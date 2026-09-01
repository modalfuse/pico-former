#include "ceqt_infer.h"
#include "rice_web.h"

#include <WiFi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#include <string.h>

#if __has_include("wifi_secrets.gen.h")
#include "wifi_secrets.gen.h"
#else
#define WIFI_SSID ""
#define WIFI_PASS ""
#endif

#ifndef PROBE_WIFI_TIMEOUT_MS
#define PROBE_WIFI_TIMEOUT_MS 20000
#endif

static const uint8_t kMagicWin[4] = {0x50, 0x46, 0x57, 0x31};
static const uint8_t kMagicRes[4] = {0x50, 0x46, 0x52, 0x31};
static const uint8_t kMagicBurn[4] = {0x50, 0x46, 0x42, 0x31};
static const uint8_t kMagicPing[4] = {0x50, 0x46, 0x50, 0x31};

enum { WIN_PAYLOAD = 2 + 1 + 1 + CEQT_N_CGM * 4 + CEQT_N_MARK * 3 * 4 + CEQT_N_MARK * 4 };

static CeqtResult g_last;
static CeqtWindow g_job;
static CeqtWindow g_burn_win;
static TaskHandle_t g_infer_task;
static SemaphoreHandle_t g_job_sem;
static SemaphoreHandle_t g_done_sem;
static portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;
static uint32_t g_seq;
static uint8_t g_kind;  // 0 idle, 1 window, 2 burn
static uint16_t g_burn_left;
static uint16_t g_burn_total;
static uint32_t g_burn_t0;

static uint8_t xor_bytes(const uint8_t *p, int n) {
    uint8_t x = 0;
    for (int i = 0; i < n; ++i) {
        x ^= p[i];
    }
    return x;
}

static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t get_u16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }

static void put_f32(uint8_t *p, float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    put_u32(p, u);
}

static float get_f32(const uint8_t *p) {
    uint32_t u = get_u32(p);
    float v;
    memcpy(&v, &u, 4);
    return v;
}

static bool start_job(const CeqtWindow *win, uint32_t seq, uint8_t kind) {
    if (g_kind) {
        return false;
    }
    memcpy(&g_job, win, sizeof(g_job));
    g_seq = seq;
    g_kind = kind;
    xSemaphoreGive(g_job_sem);
    return true;
}

static void infer_task(void *arg) {
    (void)arg;
    Serial.printf("infer on core %u\n", (unsigned)xPortGetCoreID());
    for (;;) {
        xSemaphoreTake(g_job_sem, portMAX_DELAY);
        CeqtResult tmp;
        ceqt_infer_window(&g_job, &tmp);
        tmp.infer_core = (int)xPortGetCoreID();
        portENTER_CRITICAL(&g_mux);
        memcpy(&g_last, &tmp, sizeof(g_last));
        portEXIT_CRITICAL(&g_mux);
        xSemaphoreGive(g_done_sem);
    }
}

static bool wifi_join() {
    if (WIFI_SSID[0] == '\0') {
        return false;
    }
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < PROBE_WIFI_TIMEOUT_MS) {
        delay(200);
    }
    return WiFi.status() == WL_CONNECTED;
}

static void write_result(uint32_t seq, const CeqtResult *r, uint8_t status) {
    uint8_t buf[4 + 4 + 1 + 1 + CEQT_N_EVT + 4 + 16 + CEQT_N_OUT * 4 + CEQT_N_EVT * 4 + 4];
    int n = 0;
    memcpy(buf + n, kMagicRes, 4);
    n += 4;
    put_u32(buf + n, seq);
    n += 4;
    buf[n++] = status;
    buf[n++] = (uint8_t)r->infer_core;
    for (int i = 0; i < CEQT_N_EVT; ++i) {
        buf[n++] = (uint8_t)r->evt_on[i];
    }
    put_f32(buf + n, r->g);
    n += 4;
    put_u32(buf + n, r->us_total);
    n += 4;
    put_u32(buf + n, r->us_enc);
    n += 4;
    put_u32(buf + n, r->us_cross);
    n += 4;
    put_u32(buf + n, r->us_heads);
    n += 4;
    for (int i = 0; i < CEQT_N_OUT; ++i) {
        put_f32(buf + n, r->yhat[i]);
        n += 4;
    }
    for (int i = 0; i < CEQT_N_EVT; ++i) {
        put_f32(buf + n, r->logits[i]);
        n += 4;
    }
    put_u32(buf + n, r->heap_free);
    n += 4;
    uint8_t x = xor_bytes(buf + 4, n - 4);
    Serial.write(buf, n);
    Serial.write(&x, 1);
}

static void snapshot(CeqtResult *out) {
    portENTER_CRITICAL(&g_mux);
    memcpy(out, &g_last, sizeof(*out));
    portEXIT_CRITICAL(&g_mux);
}

static void on_infer_done() {
    CeqtResult res;
    snapshot(&res);
    if (g_kind == 2) {
        if (g_burn_left > 0) {
            g_burn_left--;
        }
        if (g_burn_left > 0) {
            g_kind = 0;
            start_job(&g_burn_win, g_burn_total - g_burn_left, 2);
        } else {
            uint32_t dt = micros() - g_burn_t0;
            Serial.printf("BURN done n=%u wall_us=%u last_us=%u heap=%u\n", (unsigned)g_burn_total, (unsigned)dt,
                          (unsigned)res.us_total, (unsigned)res.heap_free);
            g_kind = 0;
        }
        return;
    }
    res.seq = g_seq;
    write_result(g_seq, &res, 0);
    g_kind = 0;
}

static int g_rx_n;
static uint8_t g_rx[8 + WIN_PAYLOAD + 8];

static void reset_rx() { g_rx_n = 0; }

static void feed_serial() {
    while (Serial.available() > 0) {
        int c = Serial.read();
        if (c < 0) {
            break;
        }
        if (g_rx_n >= (int)sizeof(g_rx)) {
            reset_rx();
        }
        g_rx[g_rx_n++] = (uint8_t)c;
        if (g_rx_n < 4) {
            const uint8_t expect_w = kMagicWin[g_rx_n - 1];
            const uint8_t expect_b = kMagicBurn[g_rx_n - 1];
            const uint8_t expect_p = kMagicPing[g_rx_n - 1];
            uint8_t got = g_rx[g_rx_n - 1];
            if (got != expect_w && got != expect_b && got != expect_p) {
                if (got == 0x50) {
                    g_rx[0] = 0x50;
                    g_rx_n = 1;
                } else {
                    reset_rx();
                }
            }
            continue;
        }
        if (g_rx_n == 4) {
            if (memcmp(g_rx, kMagicWin, 4) && memcmp(g_rx, kMagicBurn, 4) && memcmp(g_rx, kMagicPing, 4)) {
                reset_rx();
            }
            continue;
        }
        if (memcmp(g_rx, kMagicPing, 4) == 0) {
            Serial.write(kMagicPing, 4);
            reset_rx();
            continue;
        }
        if (memcmp(g_rx, kMagicBurn, 4) == 0) {
            if (g_rx_n < 6) {
                continue;
            }
            uint16_t n = get_u16(g_rx + 4);
            reset_rx();
            if (n == 0 || n > 400) {
                n = 1;
            }
            ceqt_fill_golden_window(&g_burn_win);
            g_burn_total = n;
            g_burn_left = n;
            g_burn_t0 = micros();
            Serial.printf("BURN start n=%u\n", (unsigned)n);
            if (!start_job(&g_burn_win, 0, 2)) {
                Serial.println("BURN busy");
            }
            continue;
        }
        const int need = 4 + 4 + WIN_PAYLOAD + 1;
        if (g_rx_n < need) {
            continue;
        }
        uint8_t got = g_rx[need - 1];
        uint8_t exp = xor_bytes(g_rx + 4, 4 + WIN_PAYLOAD);
        uint32_t seq = get_u32(g_rx + 4);
        CeqtResult res{};
        if (got != exp) {
            write_result(seq, &res, 2);
            reset_rx();
            continue;
        }
        CeqtWindow win{};
        const uint8_t *p = g_rx + 8;
        win.n_marks = get_u16(p);
        win.quality_ok = p[2];
        p += 4;
        for (int i = 0; i < CEQT_N_CGM; ++i) {
            win.y_hist[i] = get_f32(p);
            p += 4;
        }
        for (int i = 0; i < CEQT_N_MARK * 3; ++i) {
            win.marks[i] = get_f32(p);
            p += 4;
        }
        for (int i = 0; i < CEQT_N_MARK; ++i) {
            win.mask[i] = get_f32(p);
            p += 4;
        }
        if (!start_job(&win, seq, 1)) {
            write_result(seq, &res, 1);
        }
        reset_rx();
    }
}

void setup() {
    Serial.setRxBufferSize(1024);
    Serial.begin(115200);
    delay(300);
    Serial.println("Pico-Former rice-s3");
    if (!ceqt_init()) {
        Serial.println("alloc FAIL");
        return;
    }
    g_job_sem = xSemaphoreCreateBinary();
    g_done_sem = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(infer_task, "rice", 16384, NULL, 1, &g_infer_task, 1);

    CeqtWindow gold;
    ceqt_fill_golden_window(&gold);
    start_job(&gold, 0, 1);
    while (xSemaphoreTake(g_done_sem, pdMS_TO_TICKS(15000)) != pdTRUE) {
        delay(10);
    }
    CeqtResult first;
    snapshot(&first);
    ceqt_score_golden(&first);
    g_kind = 0;
    Serial.printf("done total=%.1f ms  dMax=%.4f  lMax=%.5f  yMax=%.4f  1h=%.1f  4h=%.1f\n", first.us_total / 1000.0,
                  first.maxabs_delta, first.maxabs_logits, first.maxabs_yhat, first.pred_1h, first.pred_4h);

    if (wifi_join()) {
        Serial.printf("wifi %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("wifi skip/fail");
    }
    if (rice_web_start(&g_last, &g_mux)) {
        Serial.printf("http://%s/\n", WiFi.localIP().toString().c_str());
    }
    Serial.println("Pico-Former ready");
}

void loop() {
    feed_serial();
    if (g_kind && xSemaphoreTake(g_done_sem, 0) == pdTRUE) {
        on_infer_done();
    }
    rice_web_loop();
    delay(1);
}
