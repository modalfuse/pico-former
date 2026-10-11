#include "ceqt_infer.h"

#include "ceqt_weights_active.gen.h"
#include "ceqt_manifest.gen.h"

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <math.h>
#include <mbedtls/sha256.h>
#include <soc/soc_memory_layout.h>
#include <string.h>

#ifndef CEQT_SEGMENT_TIMING
#define CEQT_SEGMENT_TIMING 1
#endif
#if CEQT_SEGMENT_TIMING
#define SEG_TIME() micros()
#else
#define SEG_TIME() 0u
#endif

#ifndef CEQT_GPIO_INFER
#define CEQT_GPIO_INFER 4
#endif

static volatile bool g_poison_next;

static float *g_tok;
static float *g_tmp;
static float *g_qkv;
static float *g_score;
static float *g_attn;
static float *g_curve;
static float *g_event;
static bool g_ready;

// Defined in ceqt_release.cpp; kept out of this translation unit so that the inference code is compiled unchanged.
void ceqt_release_partial(float *a, float *b, float *c, float *d, float *e, float *f, float *g);

static void linear(const float *x, const float *W, const float *b, float *y, int n, int out_d, int in_d) {
    for (int i = 0; i < n; ++i) {
        const float *xi = x + i * in_d;
        float *yo = y + i * out_d;
        for (int o = 0; o < out_d; ++o) {
            float s = b ? b[o] : 0.f;
            const float *w = W + o * in_d;
            for (int k = 0; k < in_d; ++k) {
                s += w[k] * xi[k];
            }
            yo[o] = s;
        }
    }
}

// Exact (erf) GELU, matching torch.nn.GELU() with approximate="none".
static void gelu_inplace(float *x, int n) {
    const float k = 0.70710678118654752f;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = 0.5f * v * (1.f + erff(v * k));
    }
}

static void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.f) {
            x[i] = 0.f;
        }
    }
}

static void layernorm(float *x, const float *gamma, const float *beta, int n, int d) {
    for (int i = 0; i < n; ++i) {
        float *row = x + i * d;
        float mean = 0.f;
        for (int j = 0; j < d; ++j) {
            mean += row[j];
        }
        mean /= (float)d;
        float var = 0.f;
        for (int j = 0; j < d; ++j) {
            float t = row[j] - mean;
            var += t * t;
        }
        float inv = 1.f / sqrtf(var / (float)d + 1e-5f);
        for (int j = 0; j < d; ++j) {
            row[j] = (row[j] - mean) * inv * gamma[j] + beta[j];
        }
    }
}

static void softmax_row(float *s, int n) {
    float m = s[0];
    for (int i = 1; i < n; ++i) {
        if (s[i] > m) {
            m = s[i];
        }
    }
    float z = 0.f;
    for (int i = 0; i < n; ++i) {
        s[i] = expf(s[i] - m);
        z += s[i];
    }
    float inv = 1.f / z;
    for (int i = 0; i < n; ++i) {
        s[i] *= inv;
    }
}

// q: nq x d, kv: nk x d. in_w is (3d, d).
static void mha(const float *q_src, const float *kv_src, int nq, int nk, bool self, const float *in_w,
                const float *in_b, const float *out_w, const float *out_b, float *y) {
    const int d = CEQT_D;
    const int h = CEQT_H;
    const int dk = CEQT_DK;
    const float scale = 1.f / sqrtf((float)dk);
    float *Q = g_qkv;
    float *K = g_qkv + nq * d;
    float *V = g_qkv + nq * d + nk * d;

    if (self) {
        linear(q_src, in_w, in_b, g_tmp, nq, 3 * d, d);
        for (int i = 0; i < nq; ++i) {
            memcpy(Q + i * d, g_tmp + i * 3 * d, d * sizeof(float));
            memcpy(K + i * d, g_tmp + i * 3 * d + d, d * sizeof(float));
            memcpy(V + i * d, g_tmp + i * 3 * d + 2 * d, d * sizeof(float));
        }
    } else {
        linear(q_src, in_w, in_b, Q, nq, d, d);
        linear(kv_src, in_w + d * d, in_b + d, g_tmp, nk, 2 * d, d);
        for (int i = 0; i < nk; ++i) {
            memcpy(K + i * d, g_tmp + i * 2 * d, d * sizeof(float));
            memcpy(V + i * d, g_tmp + i * 2 * d + d, d * sizeof(float));
        }
    }

    memset(y, 0, (size_t)nq * d * sizeof(float));
    for (int head = 0; head < h; ++head) {
        for (int i = 0; i < nq; ++i) {
            float *sc = g_score + i * nk;
            const float *qi = Q + i * d + head * dk;
            for (int j = 0; j < nk; ++j) {
                const float *kj = K + j * d + head * dk;
                float s = 0.f;
                for (int k = 0; k < dk; ++k) {
                    s += qi[k] * kj[k];
                }
                sc[j] = s * scale;
            }
            softmax_row(sc, nk);
            float acc[CEQT_DK];
            memset(acc, 0, sizeof(acc));
            for (int j = 0; j < nk; ++j) {
                const float *vj = V + j * d + head * dk;
                float a = sc[j];
                for (int k = 0; k < dk; ++k) {
                    acc[k] += a * vj[k];
                }
            }
            memcpy(g_tmp + i * d + head * dk, acc, sizeof(acc));
        }
    }
    linear(g_tmp, out_w, out_b, y, nq, d, d);
}

static void enc_layer(float *x, int n, const float *win, const float *bin, const float *wout, const float *bout,
                      const float *wl1, const float *bl1, const float *wl2, const float *bl2, const float *n1w,
                      const float *n1b, const float *n2w, const float *n2b) {
    mha(x, x, n, n, true, win, bin, wout, bout, g_attn);
    for (int i = 0; i < n * CEQT_D; ++i) {
        x[i] += g_attn[i];
    }
    layernorm(x, n1w, n1b, n, CEQT_D);
    linear(x, wl1, bl1, g_tmp, n, CEQT_FF, CEQT_D);
    relu_inplace(g_tmp, n * CEQT_FF);
    linear(g_tmp, wl2, bl2, g_attn, n, CEQT_D, CEQT_FF);
    for (int i = 0; i < n * CEQT_D; ++i) {
        x[i] += g_attn[i];
    }
    layernorm(x, n2w, n2b, n, CEQT_D);
}

bool ceqt_init() {
    if (g_ready) {
        return true;
    }
    const size_t tok = (size_t)CEQT_N_TOK * CEQT_D;
    const size_t tmp = (size_t)CEQT_N_TOK * 3 * CEQT_D;
    const size_t qkv = (size_t)CEQT_N_TOK * 3 * CEQT_D;
    const uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    g_tok = (float *)heap_caps_malloc(tok * sizeof(float), caps);
    g_tmp = (float *)heap_caps_malloc(tmp * sizeof(float), caps);
    g_qkv = (float *)heap_caps_malloc(qkv * sizeof(float), caps);
    g_score = (float *)heap_caps_malloc((size_t)CEQT_N_TOK * CEQT_N_TOK * sizeof(float), caps);
    g_attn = (float *)heap_caps_malloc(tok * sizeof(float), caps);
    g_curve = (float *)heap_caps_malloc((size_t)CEQT_N_OUT * CEQT_D * sizeof(float), caps);
    g_event = (float *)heap_caps_malloc((size_t)CEQT_N_EVT * CEQT_D * sizeof(float), caps);
    g_ready = g_tok && g_tmp && g_qkv && g_score && g_attn && g_curve && g_event;
    if (g_ready) {
        pinMode(CEQT_GPIO_INFER, OUTPUT);
        digitalWrite(CEQT_GPIO_INFER, LOW);
    } else {
        // Release a partial allocation so that Wi-Fi and the status interfaces can still start and report the fault.
        ceqt_release_partial(g_tok, g_tmp, g_qkv, g_score, g_attn, g_curve, g_event);
    }
    return g_ready;
}

bool ceqt_ready() { return g_ready; }

const float *ceqt_thresholds() { return W_THR; }

void ceqt_debug_poison_next() { g_poison_next = true; }

void ceqt_print_buffers() {
    const struct {
        const char *name;
        const float *p;
        size_t bytes;
    } bufs[] = {
        {"g_tok", g_tok, (size_t)CEQT_N_TOK * CEQT_D * 4},
        {"g_tmp", g_tmp, (size_t)CEQT_N_TOK * 3 * CEQT_D * 4},
        {"g_qkv", g_qkv, (size_t)CEQT_N_TOK * 3 * CEQT_D * 4},
        {"g_score", g_score, (size_t)CEQT_N_TOK * CEQT_N_TOK * 4},
        {"g_attn", g_attn, (size_t)CEQT_N_TOK * CEQT_D * 4},
        {"g_curve", g_curve, (size_t)CEQT_N_OUT * CEQT_D * 4},
        {"g_event", g_event, (size_t)CEQT_N_EVT * CEQT_D * 4},
    };
    for (const auto &b : bufs) {
        Serial.printf("BUF %s addr=%p bytes=%u internal=%d\n", b.name, (const void *)b.p, (unsigned)b.bytes,
                      (int)(b.p && esp_ptr_internal(b.p)));
    }
}

static void sha_hex(const uint8_t *d, char *out) {
    static const char *hx = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out[2 * i] = hx[d[i] >> 4];
        out[2 * i + 1] = hx[d[i] & 15];
    }
    out[64] = 0;
}

// M2: SHA-256 over the weight arrays (header order, float32 little-endian) and over the 16 thresholds,
// compared with digests computed from the source header at build time.
bool ceqt_check_integrity(bool verbose) {
    uint8_t dw[32], dt[32];
    mbedtls_sha256_context c;
    mbedtls_sha256_init(&c);
    mbedtls_sha256_starts_ret(&c, 0);
#define X(name) mbedtls_sha256_update_ret(&c, (const unsigned char *)name, sizeof(name));
    CEQT_MANIFEST_WEIGHTS
#undef X
    mbedtls_sha256_finish_ret(&c, dw);
    mbedtls_sha256_free(&c);
    mbedtls_sha256_ret((const unsigned char *)W_THR, sizeof(W_THR), dt, 0);
    const bool ok_w = memcmp(dw, CEQT_SHA256_WEIGHTS, 32) == 0;
    const bool ok_t = memcmp(dt, CEQT_SHA256_THRESHOLDS, 32) == 0;
    if (verbose) {
        char h[65];
        sha_hex(dw, h);
        Serial.printf("SHA weights %s %s\n", h, ok_w ? "ok" : "MISMATCH");
        sha_hex(dt, h);
        Serial.printf("SHA thresholds %s %s\n", h, ok_t ? "ok" : "MISMATCH");
    }
    return ok_w && ok_t;
}

static bool mark_dose_ok(int typ, float lg) {
    if (!(lg > 0.f)) {
        return false;
    }
    const float lim = typ == 0 ? CEQT_LOG1P_MEAL_MAX : (typ == 3 ? CEQT_LOG1P_BASAL_MAX : CEQT_LOG1P_BOLUS_MAX);
    return lg <= lim;
}

// R1-R8 on the node. Returns CEQT_OK or a NO_PRED_* status; invalid marks are removed from `clean`.
static uint8_t validate(const CeqtWindow *win, CeqtWindow *clean, uint8_t *n_dropped) {
    memcpy(clean, win, sizeof(*clean));
    *n_dropped = 0;
    for (int i = 0; i < CEQT_N_CGM; ++i) {
        if (!isfinite(win->y_hist[i])) {
            return CEQT_NO_PRED_INVALID;
        }
    }
    for (int i = 0; i < CEQT_N_MARK * 3; ++i) {
        if (!isfinite(win->marks[i])) {
            return CEQT_NO_PRED_INVALID;
        }
    }
    for (int i = 0; i < CEQT_N_MARK; ++i) {
        if (!isfinite(win->mask[i])) {
            return CEQT_NO_PRED_INVALID;
        }
    }
    for (int i = 0; i < CEQT_N_CGM; ++i) {
        const float y = win->y_hist[i];
        if (y < CEQT_CGM_MIN - CEQT_CGM_TOL || y > CEQT_CGM_MAX + CEQT_CGM_TOL) {
            return CEQT_NO_PRED_GAP;
        }
        if (y < CEQT_CGM_MIN || y > CEQT_CGM_MAX) {
            clean->y_hist[i] = y < CEQT_CGM_MIN ? CEQT_CGM_MIN : CEQT_CGM_MAX;
            clean->flags |= CEQT_FLAG_CGM_CLAMPED;
        }
    }
    if (win->t0_s && win->t_last_s && (win->t0_s < win->t_last_s || win->t0_s - win->t_last_s > CEQT_STALE_INPUT_S)) {
        return CEQT_NO_PRED_STALE;
    }
    for (int j = 0; j < CEQT_N_MARK; ++j) {
        if (win->mask[j] == 0.f) {
            continue;
        }
        const float typf = win->marks[j * 3 + 0];
        const int typ = (int)typf;
        const bool ok = typf == (float)typ && typ >= 0 && typ <= 3 && mark_dose_ok(typ, win->marks[j * 3 + 1]) &&
                        win->marks[j * 3 + 2] >= 0.f && win->mask[j] == 1.f;
        if (!ok) {
            clean->mask[j] = 0.f;
            clean->marks[j * 3 + 0] = clean->marks[j * 3 + 1] = clean->marks[j * 3 + 2] = 0.f;
            (*n_dropped)++;
        }
    }
    if (*n_dropped) {
        clean->quality_ok = 0;
        clean->flags |= CEQT_FLAG_MARKS_DROPPED;
    }
    return CEQT_OK;
}

static float maxabs_diff(const float *a, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static float last_finite(const float *y, int n) {
    float last = y[0];
    for (int i = 0; i < n; ++i) {
        if (isfinite(y[i])) {
            last = y[i];
        }
    }
    return last;
}

void ceqt_fill_golden_window(CeqtWindow *win) {
    memset(win, 0, sizeof(*win));
    memcpy(win->y_hist, G_Y_HIST, sizeof(win->y_hist));
    memcpy(win->marks, G_MARKS, sizeof(win->marks));
    memcpy(win->mask, G_MASK, sizeof(win->mask));
    win->n_marks = (uint16_t)CEQT_N_MARKS;
    win->quality_ok = 1;
}

static void clear_events(CeqtResult *out) {
    for (int i = 0; i < CEQT_N_EVT; ++i) {
        out->evt_on[i] = 0;
    }
}

static void infer_valid(const CeqtWindow *win, CeqtResult *out);

void ceqt_infer_window(const CeqtWindow *raw, CeqtResult *out) {
    memset(out, 0, sizeof(*out));
    out->infer_core = (int)xPortGetCoreID();
    out->t0_s = raw->t0_s;
    out->flags = raw->flags & CEQT_FLAG_MARKS_TRUNCATED;
    memcpy(out->y_hist, raw->y_hist, sizeof(out->y_hist));
    if (!g_ready) {
        out->status = CEQT_FAULT_INFER;
        return;
    }
    CeqtWindow win;
    out->status = validate(raw, &win, &out->n_dropped);
    out->flags |= win.flags & (CEQT_FLAG_MARKS_DROPPED | CEQT_FLAG_CGM_CLAMPED);
    if (out->status != CEQT_OK) {
        return;
    }
    digitalWrite(CEQT_GPIO_INFER, HIGH);
    infer_valid(&win, out);
    digitalWrite(CEQT_GPIO_INFER, LOW);
    if (g_poison_next) {
        g_poison_next = false;
        out->logits[0] = NAN;
    }
    bool ok = true;
    for (int i = 0; i < CEQT_N_EVT; ++i) {
        ok = ok && isfinite(out->logits[i]);
    }
    for (int i = 0; i < CEQT_N_OUT; ++i) {
        ok = ok && isfinite(out->yhat[i]) && out->yhat[i] >= 0.f && out->yhat[i] <= CEQT_YHAT_MAX;
    }
    if (!ok) {
        out->status = CEQT_FAULT_OUTPUT;
        clear_events(out);
    }
}

static void infer_valid(const CeqtWindow *win, CeqtResult *out) {
    uint32_t t0 = micros();

    for (int i = 0; i < CEQT_N_CGM; ++i) {
        float yn = (win->y_hist[i] - CEQT_Y_MEAN) / CEQT_Y_STD;
        for (int d = 0; d < CEQT_D; ++d) {
            g_tok[i * CEQT_D + d] = W_CGM[d] * yn + B_CGM[d] + W_POS[i * CEQT_D + d];
        }
    }
    for (int j = 0; j < CEQT_N_MARK; ++j) {
        int typ = (int)win->marks[j * 3 + 0];
        if (typ < 0) {
            typ = 0;
        }
        if (typ > 3) {
            typ = 3;
        }
        float feat[2] = {win->marks[j * 3 + 1], win->marks[j * 3 + 2]};
        float h[CEQT_D];
        linear(feat, W_MARK1, B_MARK1, h, 1, CEQT_D, 2);
        gelu_inplace(h, CEQT_D);
        float m2[CEQT_D];
        linear(h, W_MARK2, B_MARK2, m2, 1, CEQT_D, CEQT_D);
        float mk = win->mask[j];
        float *dst = g_tok + (CEQT_N_CGM + j) * CEQT_D;
        for (int d = 0; d < CEQT_D; ++d) {
            dst[d] = (W_TYPE[typ * CEQT_D + d] + m2[d]) * mk;
        }
    }

    uint32_t t1 = SEG_TIME();
    enc_layer(g_tok, CEQT_N_TOK, W_E0_IN, B_E0_IN, W_E0_OUT, B_E0_OUT, W_E0_L1, B_E0_L1, W_E0_L2, B_E0_L2, W_E0_N1,
              B_E0_N1, W_E0_N2, B_E0_N2);
    enc_layer(g_tok, CEQT_N_TOK, W_E1_IN, B_E1_IN, W_E1_OUT, B_E1_OUT, W_E1_L1, B_E1_L1, W_E1_L2, B_E1_L2, W_E1_N1,
              B_E1_N1, W_E1_N2, B_E1_N2);
    uint32_t t2 = SEG_TIME();

    mha(W_CURVE_Q, g_tok, CEQT_N_OUT, CEQT_N_TOK, false, W_CROSS_IN, B_CROSS_IN, W_CROSS_OUT, B_CROSS_OUT, g_curve);
    mha(W_EVENT_Q, g_tok, CEQT_N_EVT, CEQT_N_TOK, false, W_CROSS_IN, B_CROSS_IN, W_CROSS_OUT, B_CROSS_OUT, g_event);
    uint32_t t3 = SEG_TIME();

    linear(g_curve, W_DELTA1, B_DELTA1, g_attn, CEQT_N_OUT, CEQT_D, CEQT_D);
    gelu_inplace(g_attn, CEQT_N_OUT * CEQT_D);
    linear(g_attn, W_DELTA2, B_DELTA2, out->delta, CEQT_N_OUT, 1, CEQT_D);
    for (int i = 0; i < CEQT_N_OUT; ++i) {
        out->delta[i] = 40.f * tanhf(out->delta[i] / 40.f) * CEQT_Y_STD;
    }

    linear(g_event, W_EVT1, B_EVT1, g_tmp, CEQT_N_EVT, CEQT_D, CEQT_D);
    gelu_inplace(g_tmp, CEQT_N_EVT * CEQT_D);
    linear(g_tmp, W_EVT2, B_EVT2, out->logits, CEQT_N_EVT, 1, CEQT_D);

    // INK defaults: a_last=1, a_level=0, a_slope=0 → last finite CGM.
    float last = last_finite(win->y_hist, CEQT_N_CGM);
    out->g = (win->n_marks > 0) ? (win->quality_ok ? 1.f : 0.45f) : 0.f;
    for (int i = 0; i < CEQT_N_OUT; ++i) {
        out->y_iner[i] = last;
        out->yhat[i] = out->y_iner[i] + out->g * out->delta[i];
    }
    out->now_mgdl = last;
    out->pred_1h = out->yhat[11];
    out->pred_4h = out->yhat[47];
    for (int i = 0; i < CEQT_N_EVT; ++i) {
        out->evt_on[i] = out->logits[i] >= W_THR[i] ? 1 : 0;
    }
    out->heap_free = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    uint32_t t4 = micros();
#if CEQT_SEGMENT_TIMING
    out->us_enc = t2 - t1;
    out->us_cross = t3 - t2;
    out->us_heads = t4 - t3;
#else
    (void)t1;
    (void)t2;
    (void)t3;
#endif
    out->us_total = t4 - t0;
}

void ceqt_score_golden(CeqtResult *out) {
    out->maxabs_delta = maxabs_diff(out->delta, G_DELTA, CEQT_N_OUT);
    out->maxabs_logits = maxabs_diff(out->logits, G_LOGITS, CEQT_N_EVT);
    out->maxabs_yhat = maxabs_diff(out->yhat, G_YHAT, CEQT_N_OUT);
}

void ceqt_infer_golden(CeqtResult *out) {
    CeqtWindow win;
    ceqt_fill_golden_window(&win);
    ceqt_infer_window(&win, out);
    ceqt_score_golden(out);
}
