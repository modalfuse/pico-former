// R2-15 board latency benchmark. Kernels, buffer layout and allocation flags follow
// pico-former/firmware/src/ceqt_infer.cpp; width, depth, heads and token count come from bench_select.h.
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <math.h>
#include <string.h>

#include "bench_select.h"

#define D BD
#define H BH
#define DK (BD / BH)
#define FF (2 * BD)
#define N_TOK (BN_CGM + BN_MARK)
#define N_OUT 48
#define N_EVT 16
#define N_WARM 3
#define N_RUN 80

struct EncW {
    const float *win, *bin, *wout, *bout, *wl1, *bl1, *wl2, *bl2, *n1w, *n1b, *n2w, *n2b;
};
static const EncW ENC[BL] = BENCH_ENC_INIT;

static float *g_tok, *g_tmp, *g_qkv, *g_score, *g_attn, *g_curve, *g_event;
static float g_y[BN_CGM], g_marks[BN_MARK * 3], g_mask[BN_MARK];
static float g_delta[N_OUT], g_logits[N_EVT];

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

static void mha(const float *q_src, const float *kv_src, int nq, int nk, bool self, const float *in_w,
                const float *in_b, const float *out_w, const float *out_b, float *y) {
    const float scale = 1.f / sqrtf((float)DK);
    float *Q = g_qkv;
    float *K = g_qkv + nq * D;
    float *V = g_qkv + nq * D + nk * D;
    if (self) {
        linear(q_src, in_w, in_b, g_tmp, nq, 3 * D, D);
        for (int i = 0; i < nq; ++i) {
            memcpy(Q + i * D, g_tmp + i * 3 * D, D * sizeof(float));
            memcpy(K + i * D, g_tmp + i * 3 * D + D, D * sizeof(float));
            memcpy(V + i * D, g_tmp + i * 3 * D + 2 * D, D * sizeof(float));
        }
    } else {
        linear(q_src, in_w, in_b, Q, nq, D, D);
        linear(kv_src, in_w + D * D, in_b + D, g_tmp, nk, 2 * D, D);
        for (int i = 0; i < nk; ++i) {
            memcpy(K + i * D, g_tmp + i * 2 * D, D * sizeof(float));
            memcpy(V + i * D, g_tmp + i * 2 * D + D, D * sizeof(float));
        }
    }
    memset(y, 0, (size_t)nq * D * sizeof(float));
    for (int head = 0; head < H; ++head) {
        for (int i = 0; i < nq; ++i) {
            float *sc = g_score + i * nk;
            const float *qi = Q + i * D + head * DK;
            for (int j = 0; j < nk; ++j) {
                const float *kj = K + j * D + head * DK;
                float s = 0.f;
                for (int k = 0; k < DK; ++k) {
                    s += qi[k] * kj[k];
                }
                sc[j] = s * scale;
            }
            softmax_row(sc, nk);
            float acc[DK];
            memset(acc, 0, sizeof(acc));
            for (int j = 0; j < nk; ++j) {
                const float *vj = V + j * D + head * DK;
                float a = sc[j];
                for (int k = 0; k < DK; ++k) {
                    acc[k] += a * vj[k];
                }
            }
            memcpy(g_tmp + i * D + head * DK, acc, sizeof(acc));
        }
    }
    linear(g_tmp, out_w, out_b, y, nq, D, D);
}

static void enc_layer(float *x, int n, const EncW &w) {
    mha(x, x, n, n, true, w.win, w.bin, w.wout, w.bout, g_attn);
    for (int i = 0; i < n * D; ++i) {
        x[i] += g_attn[i];
    }
    layernorm(x, w.n1w, w.n1b, n, D);
    linear(x, w.wl1, w.bl1, g_tmp, n, FF, D);
    relu_inplace(g_tmp, n * FF);
    linear(g_tmp, w.wl2, w.bl2, g_attn, n, D, FF);
    for (int i = 0; i < n * D; ++i) {
        x[i] += g_attn[i];
    }
    layernorm(x, w.n2w, w.n2b, n, D);
}

static void infer(uint32_t *us_total, uint32_t *us_enc, uint32_t *us_cross, uint32_t *us_heads) {
    uint32_t t0 = micros();
    for (int i = 0; i < BN_CGM; ++i) {
        float yn = (g_y[i] - 149.09941f) / 49.303144f;
        for (int d = 0; d < D; ++d) {
            g_tok[i * D + d] = W_CGM[d] * yn + B_CGM[d] + W_POS[i * D + d];
        }
    }
    for (int j = 0; j < BN_MARK; ++j) {
        int typ = (int)g_marks[j * 3 + 0];
        float feat[2] = {g_marks[j * 3 + 1], g_marks[j * 3 + 2]};
        float h[D];
        linear(feat, W_MARK1, B_MARK1, h, 1, D, 2);
        gelu_inplace(h, D);
        float m2[D];
        linear(h, W_MARK2, B_MARK2, m2, 1, D, D);
        float *dst = g_tok + (BN_CGM + j) * D;
        for (int d = 0; d < D; ++d) {
            dst[d] = (W_TYPE[typ * D + d] + m2[d]) * g_mask[j];
        }
    }
    uint32_t t1 = micros();
    for (int l = 0; l < BL; ++l) {
        enc_layer(g_tok, N_TOK, ENC[l]);
    }
    uint32_t t2 = micros();
    mha(W_CURVE_Q, g_tok, N_OUT, N_TOK, false, W_CROSS_IN, B_CROSS_IN, W_CROSS_OUT, B_CROSS_OUT, g_curve);
    mha(W_EVENT_Q, g_tok, N_EVT, N_TOK, false, W_CROSS_IN, B_CROSS_IN, W_CROSS_OUT, B_CROSS_OUT, g_event);
    uint32_t t3 = micros();
    linear(g_curve, W_DELTA1, B_DELTA1, g_attn, N_OUT, D, D);
    gelu_inplace(g_attn, N_OUT * D);
    linear(g_attn, W_DELTA2, B_DELTA2, g_delta, N_OUT, 1, D);
    for (int i = 0; i < N_OUT; ++i) {
        g_delta[i] = 40.f * tanhf(g_delta[i] / 40.f) * 49.303144f;
    }
    linear(g_event, W_EVT1, B_EVT1, g_tmp, N_EVT, D, D);
    gelu_inplace(g_tmp, N_EVT * D);
    linear(g_tmp, W_EVT2, B_EVT2, g_logits, N_EVT, 1, D);
    uint32_t t4 = micros();
    *us_total = t4 - t0;
    *us_enc = t2 - t1;
    *us_cross = t3 - t2;
    *us_heads = t4 - t3;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static uint32_t median(uint32_t *v, int n) {
    qsort(v, n, sizeof(uint32_t), cmp_u32);
    return v[n / 2];
}

void setup() {
    Serial.begin(460800);
    delay(1500);
    const size_t act = (size_t)4 * (8 * N_TOK * D + N_TOK * N_TOK + 64 * D);
    const uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    const size_t free0 = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    const size_t big0 = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    g_tok = (float *)heap_caps_malloc((size_t)N_TOK * D * 4, caps);
    g_tmp = (float *)heap_caps_malloc((size_t)N_TOK * 3 * D * 4, caps);
    g_qkv = (float *)heap_caps_malloc((size_t)N_TOK * 3 * D * 4, caps);
    g_score = (float *)heap_caps_malloc((size_t)N_TOK * N_TOK * 4, caps);
    g_attn = (float *)heap_caps_malloc((size_t)N_TOK * D * 4, caps);
    g_curve = (float *)heap_caps_malloc((size_t)N_OUT * D * 4, caps);
    g_event = (float *)heap_caps_malloc((size_t)N_EVT * D * 4, caps);
    const bool ok = g_tok && g_tmp && g_qkv && g_score && g_attn && g_curve && g_event;
    Serial.printf("BENCH cfg=%s d=%d layers=%d heads=%d n_tok=%d params=%d act_bytes=%u\n", BENCH_NAME, D, BL, H,
                  N_TOK, BENCH_N_PARAM, (unsigned)act);
    Serial.printf("BENCH heap_internal_before free=%u largest=%u after free=%u largest=%u alloc_ok=%d\n",
                  (unsigned)free0, (unsigned)big0, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL), (int)ok);
    if (!ok) {
        Serial.println("BENCH_DONE alloc_failed");
        return;
    }
    for (int i = 0; i < BN_CGM; ++i) {
        g_y[i] = 140.f + 40.f * sinf(0.08f * i);
    }
    for (int j = 0; j < BN_MARK; ++j) {
        g_marks[j * 3 + 0] = (float)(j % 4);
        g_marks[j * 3 + 1] = 1.f + 0.1f * j;
        g_marks[j * 3 + 2] = 0.05f * j;
        g_mask[j] = j < 4 ? 1.f : 0.f;
    }
    static uint32_t tt[N_RUN], te[N_RUN], tc[N_RUN], th[N_RUN];
    uint32_t a, b, c, d;
    for (int i = 0; i < N_WARM; ++i) {
        infer(&a, &b, &c, &d);
        vTaskDelay(1);
    }
    for (int i = 0; i < N_RUN; ++i) {
        infer(&tt[i], &te[i], &tc[i], &th[i]);
        vTaskDelay(1);
    }
    uint32_t mn = tt[0], mx = tt[0];
    for (int i = 1; i < N_RUN; ++i) {
        mn = tt[i] < mn ? tt[i] : mn;
        mx = tt[i] > mx ? tt[i] : mx;
    }
    Serial.printf("BENCH logits0=%.6f delta0=%.6f\n", g_logits[0], g_delta[0]);
    Serial.printf("BENCH n=%d median_us total=%u enc=%u cross=%u heads=%u min=%u max=%u\n", N_RUN,
                  (unsigned)median(tt, N_RUN), (unsigned)median(te, N_RUN), (unsigned)median(tc, N_RUN),
                  (unsigned)median(th, N_RUN), (unsigned)mn, (unsigned)mx);
    Serial.println("BENCH_DONE ok");
}

void loop() { delay(1000); }
