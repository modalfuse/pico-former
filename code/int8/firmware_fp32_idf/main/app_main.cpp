// R2-14 FP32-B: the FP32 kernels of pico-former/firmware/src/ceqt_infer.cpp (exact GELU, same weights header),
// built with ESP-IDF 5.5.3 and the same sdkconfig as firmware_espdl_full, so that FP32 and INT8 differ only in
// number format and kernel library.
// Serial protocol and timed span are those of firmware_espdl_full:
//   host -> board: "QTK1" | u32 seq | f32[96*48] tokens | u8 xor(payload)
//   board -> host: "QRS1" | u32 seq | u32 us_run | f32[48] delta_norm | f32[16] logits | u32 int_free | u32 int_min
//                  | u32 psram_free | u8 xor(payload)

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ceqt_weights.h"
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

enum {
    CEQT_N_TOK = 96,
    CEQT_D = 48,
    CEQT_H = 4,
    CEQT_DK = 12,
    CEQT_N_OUT = 48,
    CEQT_N_EVT = 16,
    CEQT_FF = 96,
};
static constexpr int TOK_BYTES = CEQT_N_TOK * CEQT_D * 4;
static constexpr uart_port_t PORT = UART_NUM_0;

static float *g_tok, *g_tmp, *g_qkv, *g_score, *g_attn, *g_curve, *g_event;

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

// Encoder, cross-attention and both heads on a 96x48 token matrix in g_tok; delta is the limited,
// standardised residual 40*tanh(d/40) before scaling by CEQT_Y_STD.
static void run_graph(float *delta, float *logits) {
    enc_layer(g_tok, CEQT_N_TOK, W_E0_IN, B_E0_IN, W_E0_OUT, B_E0_OUT, W_E0_L1, B_E0_L1, W_E0_L2, B_E0_L2, W_E0_N1,
              B_E0_N1, W_E0_N2, B_E0_N2);
    enc_layer(g_tok, CEQT_N_TOK, W_E1_IN, B_E1_IN, W_E1_OUT, B_E1_OUT, W_E1_L1, B_E1_L1, W_E1_L2, B_E1_L2, W_E1_N1,
              B_E1_N1, W_E1_N2, B_E1_N2);
    mha(W_CURVE_Q, g_tok, CEQT_N_OUT, CEQT_N_TOK, false, W_CROSS_IN, B_CROSS_IN, W_CROSS_OUT, B_CROSS_OUT, g_curve);
    mha(W_EVENT_Q, g_tok, CEQT_N_EVT, CEQT_N_TOK, false, W_CROSS_IN, B_CROSS_IN, W_CROSS_OUT, B_CROSS_OUT, g_event);
    linear(g_curve, W_DELTA1, B_DELTA1, g_attn, CEQT_N_OUT, CEQT_D, CEQT_D);
    gelu_inplace(g_attn, CEQT_N_OUT * CEQT_D);
    linear(g_attn, W_DELTA2, B_DELTA2, delta, CEQT_N_OUT, 1, CEQT_D);
    for (int i = 0; i < CEQT_N_OUT; ++i) {
        delta[i] = 40.f * tanhf(delta[i] / 40.f);
    }
    linear(g_event, W_EVT1, B_EVT1, g_tmp, CEQT_N_EVT, CEQT_D, CEQT_D);
    gelu_inplace(g_tmp, CEQT_N_EVT * CEQT_D);
    linear(g_tmp, W_EVT2, B_EVT2, logits, CEQT_N_EVT, 1, CEQT_D);
}

static bool read_exact(uint8_t *dst, int n) {
    int got = 0;
    while (got < n) {
        const int r = uart_read_bytes(PORT, dst + got, n - got, pdMS_TO_TICKS(2000));
        if (r <= 0) return false;
        got += r;
    }
    return true;
}

static uint8_t xor_of(const uint8_t *p, int n) {
    uint8_t x = 0;
    for (int i = 0; i < n; ++i) x ^= p[i];
    return x;
}

extern "C" void app_main(void) {
    uart_driver_install(PORT, 40 * 1024, 4 * 1024, 0, nullptr, 0);
    printf("FP32-B CEQT graph S3 (ESP-IDF)\n");
    const size_t free0 = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    const uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    const size_t tok = (size_t)CEQT_N_TOK * CEQT_D * 4;
    g_tok = (float *)heap_caps_malloc(tok, caps);
    g_tmp = (float *)heap_caps_malloc(3 * tok, caps);
    g_qkv = (float *)heap_caps_malloc(3 * tok, caps);
    g_score = (float *)heap_caps_malloc((size_t)CEQT_N_TOK * CEQT_N_TOK * 4, caps);
    g_attn = (float *)heap_caps_malloc(tok, caps);
    g_curve = (float *)heap_caps_malloc((size_t)CEQT_N_OUT * CEQT_D * 4, caps);
    g_event = (float *)heap_caps_malloc((size_t)CEQT_N_EVT * CEQT_D * 4, caps);
    if (!(g_tok && g_tmp && g_qkv && g_score && g_attn && g_curve && g_event)) {
        printf("alloc FAIL\n");
        vTaskDelay(portMAX_DELAY);
    }
    printf("act_alloc=%u heap_internal=%u heap_internal_min=%u heap_psram=%u\n",
           (unsigned)(free0 - heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    printf("QTK ready\n");
    fflush(stdout);

    static uint8_t frame[4 + TOK_BYTES + 1];
    static uint8_t reply[4 + 4 + 4 + 4 * (CEQT_N_OUT + CEQT_N_EVT) + 12 + 1];
    float delta[CEQT_N_OUT], logits[CEQT_N_EVT];
    uint8_t m[4];
    while (true) {
        if (!read_exact(m, 1) || m[0] != 'Q') continue;
        if (!read_exact(m + 1, 3) || memcmp(m, "QTK1", 4) != 0) continue;
        if (!read_exact(frame, sizeof(frame))) continue;
        if (xor_of(frame, 4 + TOK_BYTES) != frame[4 + TOK_BYTES]) {
            const char nak[] = "QNAK";
            uart_write_bytes(PORT, nak, 4);
            continue;
        }
        const int64_t t0 = esp_timer_get_time();
        memcpy(g_tok, frame + 4, TOK_BYTES);
        run_graph(delta, logits);
        const uint32_t us = (uint32_t)(esp_timer_get_time() - t0);
        uint8_t *p = reply;
        memcpy(p, "QRS1", 4); p += 4;
        memcpy(p, frame, 4); p += 4;
        memcpy(p, &us, 4); p += 4;
        memcpy(p, delta, 4 * CEQT_N_OUT); p += 4 * CEQT_N_OUT;
        memcpy(p, logits, 4 * CEQT_N_EVT); p += 4 * CEQT_N_EVT;
        const uint32_t mem[3] = {(uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                                 (uint32_t)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
                                 (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM)};
        memcpy(p, mem, 12); p += 12;
        *p = xor_of(reply + 4, (int)(p - reply - 4)); ++p;
        uart_write_bytes(PORT, reply, p - reply);
    }
}
