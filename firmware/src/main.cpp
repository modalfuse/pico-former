#include "ceqt_infer.h"
#include "rice_web.h"

#include <Preferences.h>
#include <WiFi.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <esp_task_wdt.h>
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
#ifndef AUDIT_BAUD
#define AUDIT_BAUD 921600
#endif
#ifndef DEPLOY_BUILD
#define DEPLOY_BUILD 0
#endif
#ifndef INFER_TIMEOUT_MS
#define INFER_TIMEOUT_MS 2000
#endif
#ifndef INTEGRITY_PERIOD_S
#define INTEGRITY_PERIOD_S 3600
#endif

static const uint8_t kMagicWin1[4] = {'P', 'F', 'W', '1'};
static const uint8_t kMagicWin2[4] = {'P', 'F', 'W', '2'};
static const uint8_t kMagicRes[4] = {'P', 'F', 'R', '2'};
static const uint8_t kMagicBurn[4] = {'P', 'F', 'B', '1'};
static const uint8_t kMagicPing[4] = {'P', 'F', 'P', '1'};
static const uint8_t kMagicDbg[4] = {'P', 'F', 'D', '1'};

enum {
    WIN1_PAYLOAD = 2 + 1 + 1 + CEQT_N_CGM * 4 + CEQT_N_MARK * 3 * 4 + CEQT_N_MARK * 4,
    WIN2_PAYLOAD = WIN1_PAYLOAD + 8,
};

// ---- shared state ----
static CeqtWindow g_job;
static CeqtResult g_res;
static TaskHandle_t g_infer_task, g_serial_task, g_sup_task;
static SemaphoreHandle_t g_job_sem, g_done_sem, g_job_lock;
static volatile uint8_t g_fault;  // 0 or CEQT_FAULT_*
static volatile bool g_infer_stuck;
static volatile bool g_debug_hang;
static volatile bool g_sha_ok = true;
static volatile uint32_t g_sha_checks;
static uint8_t g_reset_reason;
static uint16_t g_reset_count;
static uint8_t g_lmode;

static volatile uint8_t g_link;  // 0 down, 1 up
static volatile uint16_t g_reconnects;
static volatile uint32_t g_down_since_ms, g_down_total_ms, g_last_down_ms;
static volatile bool g_web_started; TEST_HOOK_GLOBALS

// ---- helpers ----
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
static void put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
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

// ---- inference (Core 1) ----
static void infer_task(void *arg) {
    (void)arg;
    esp_task_wdt_add(NULL);
    Serial.printf("infer on core %u\n", (unsigned)xPortGetCoreID());
    for (;;) {
        esp_task_wdt_reset();
        if (xSemaphoreTake(g_job_sem, pdMS_TO_TICKS(1000)) != pdTRUE) {
            continue;
        }
        if (g_debug_hang) {
            for (;;) {
            }
        }
        ceqt_infer_window(&g_job, &g_res);
        g_res.infer_core = (int)xPortGetCoreID();
        g_res.done_ms = millis();
        xSemaphoreGive(g_done_sem);
    }
}

// Runs one window to completion; returns false on timeout (M3).
static bool run_job(const CeqtWindow *win, uint32_t seq, CeqtResult *out, uint32_t timeout_ms) {
    memset(out, 0, sizeof(*out));
    out->seq = seq;
    out->t0_s = win->t0_s;
    out->done_ms = millis();
    if (g_fault) {
        out->status = g_fault;
        return true;
    }
    if (g_infer_stuck) {
        if (xSemaphoreTake(g_done_sem, 0) == pdTRUE) {
            g_infer_stuck = false;
        } else {
            out->status = CEQT_FAULT_INFER;
            return true;
        }
    }
    xSemaphoreTake(g_job_lock, portMAX_DELAY);
    memcpy(&g_job, win, sizeof(g_job));
    xSemaphoreGive(g_job_sem);
    const bool done = xSemaphoreTake(g_done_sem, pdMS_TO_TICKS(timeout_ms)) == pdTRUE;
    if (done) {
        memcpy(out, &g_res, sizeof(*out));
    } else {
        g_infer_stuck = true;
        out->status = CEQT_FAULT_INFER;
        out->done_ms = millis();
    }
    out->seq = seq;
    xSemaphoreGive(g_job_lock);
    return done;
}

// ---- status ----
static const char *link_name() { return g_link ? "up" : "down"; }

size_t node_status_json(char *buf, size_t n) {
    NetStats ns;
    rice_web_stats(&ns);
    const uint32_t now = millis();
    const uint32_t down_now = g_link ? 0 : now - g_down_since_ms;
    return (size_t)snprintf(
        buf, n,
        "{\"uptime_s\":%u,\"fault\":%u,\"reset_reason\":%u,\"reset_count\":%u,\"sha_ok\":%d,\"sha_checks\":%u,"
        "\"link\":\"%s\",\"rssi\":%d,\"reconnects\":%u,\"down_now_ms\":%u,\"down_total_ms\":%u,\"last_down_ms\":%u,"
        "\"int_free\":%u,\"int_min\":%u,\"int_largest\":%u,\"spiram_free\":%u,"
        "\"stack_infer\":%u,\"stack_serial\":%u,\"stack_sup\":%u,"
        "\"http_req\":%u,\"http_fail\":%u,\"https_req\":%u,\"https_fail\":%u,\"handler_us_max\":%u,"
        "\"http_stack_hw\":%u,\"https_stack_hw\":%u,\"stale_ms\":%u,\"lmode\":%u}",
        (unsigned)(now / 1000), (unsigned)g_fault, (unsigned)g_reset_reason, (unsigned)g_reset_count, g_sha_ok ? 1 : 0,
        (unsigned)g_sha_checks, link_name(), g_link ? (int)WiFi.RSSI() : 0, (unsigned)g_reconnects,
        (unsigned)down_now, (unsigned)g_down_total_ms, (unsigned)g_last_down_ms,
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        g_infer_task ? (unsigned)uxTaskGetStackHighWaterMark(g_infer_task) : 0u,
        g_serial_task ? (unsigned)uxTaskGetStackHighWaterMark(g_serial_task) : 0u,
        g_sup_task ? (unsigned)uxTaskGetStackHighWaterMark(g_sup_task) : 0u, (unsigned)ns.http_req,
        (unsigned)ns.http_fail, (unsigned)ns.https_req, (unsigned)ns.https_fail, (unsigned)ns.handler_us_max,
        (unsigned)ns.http_stack_hw, (unsigned)ns.https_stack_hw, (unsigned)rice_web_stale_ms(), (unsigned)g_lmode);
}

static void print_heap(const char *stage) {
    Serial.printf("HEAP %s int_free=%u int_largest=%u int_min=%u spiram_free=%u\n", stage,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

// ---- serial audit protocol (Core 0) ----
enum { RES_BYTES = 4 + 4 + 4 + CEQT_N_EVT + 4 + 16 + CEQT_N_OUT * 4 + CEQT_N_EVT * 4 + 4 + 24 + 4 + 16 + 3 };

static void write_result(const CeqtResult *r) {
    uint8_t buf[RES_BYTES + 1];
    int n = 0;
    memcpy(buf, kMagicRes, 4);
    n += 4;
    put_u32(buf + n, r->seq);
    n += 4;
    buf[n++] = r->status;
    buf[n++] = r->flags;
    buf[n++] = (uint8_t)r->infer_core;
    buf[n++] = r->n_dropped;
    for (int i = 0; i < CEQT_N_EVT; ++i) {
        buf[n++] = r->status == CEQT_OK ? (uint8_t)r->evt_on[i] : 0xFF;
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
    for (int i = 0; i < CEQT_N_OUT; ++i, n += 4) {
        put_f32(buf + n, r->yhat[i]);
    }
    for (int i = 0; i < CEQT_N_EVT; ++i, n += 4) {
        put_f32(buf + n, r->logits[i]);
    }
    put_u32(buf + n, r->t0_s);
    n += 4;
    const uint32_t mem[6] = {
        (uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
        (uint32_t)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
        (uint32_t)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
        (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        g_infer_task ? (uint32_t)uxTaskGetStackHighWaterMark(g_infer_task) : 0u,
        (uint32_t)uxTaskGetStackHighWaterMark(NULL),
    };
    for (int i = 0; i < 6; ++i, n += 4) {
        put_u32(buf + n, mem[i]);
    }
    buf[n++] = (uint8_t)(int8_t)(g_link ? WiFi.RSSI() : 0);
    buf[n++] = g_link;
    put_u16(buf + n, g_reconnects);
    n += 2;
    NetStats ns;
    rice_web_stats(&ns);
    put_u32(buf + n, ns.http_req);
    n += 4;
    put_u32(buf + n, ns.http_fail);
    n += 4;
    put_u32(buf + n, ns.https_req);
    n += 4;
    put_u32(buf + n, ns.https_fail);
    n += 4;
    buf[n++] = g_reset_reason;
    put_u16(buf + n, g_reset_count);
    n += 2;
    buf[n] = xor_bytes(buf + 4, n - 4);
    Serial.write(buf, n + 1);
}

static void finish(CeqtResult *r) {
    write_result(r);
    rice_web_publish(r);
}

static void parse_window(const uint8_t *p, bool v2, CeqtWindow *win) {
    memset(win, 0, sizeof(*win));
    win->n_marks = get_u16(p);
    win->quality_ok = p[2];
    win->flags = p[3];
    p += 4;
    if (v2) {
        win->t0_s = get_u32(p);
        win->t_last_s = get_u32(p + 4);
        p += 8;
    }
    for (int i = 0; i < CEQT_N_CGM; ++i, p += 4) {
        win->y_hist[i] = get_f32(p);
    }
    for (int i = 0; i < CEQT_N_MARK * 3; ++i, p += 4) {
        win->marks[i] = get_f32(p);
    }
    for (int i = 0; i < CEQT_N_MARK; ++i, p += 4) {
        win->mask[i] = get_f32(p);
    }
}

static void do_burn(uint16_t n) {
    CeqtWindow gold;
    ceqt_fill_golden_window(&gold);
    CeqtResult r;
    Serial.printf("BURN start n=%u\n", (unsigned)n);
    const uint32_t t0 = micros();
    uint32_t sum = 0, mn = 0xFFFFFFFF, mx = 0;
    for (uint16_t i = 0; i < n; ++i) {
        run_job(&gold, i, &r, INFER_TIMEOUT_MS);
        sum += r.us_total;
        mn = r.us_total < mn ? r.us_total : mn;
        mx = r.us_total > mx ? r.us_total : mx;
        esp_task_wdt_reset();
    }
    Serial.printf("BURN done n=%u wall_us=%u last_us=%u min_us=%u max_us=%u mean_us=%u int_free=%u\n", (unsigned)n,
                  (unsigned)(micros() - t0), (unsigned)r.us_total, (unsigned)mn, (unsigned)mx,
                  (unsigned)(n ? sum / n : 0), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
}

static void handle_debug(uint8_t cmd, uint32_t arg) {
    Preferences prefs;
    switch (cmd) {
        case 1:
            ceqt_debug_poison_next();
            break;
        case 2:
            g_debug_hang = true;
            break;
        case 3:
            rice_web_set_stale_ms(arg);
            break;
        case 4:
            break;
        case 5:
            g_sha_ok = ceqt_check_integrity(true);
            g_sha_checks++;
            if (!g_sha_ok) {
                g_fault = CEQT_FAULT_WEIGHTS;
            }
            break;
        case 6:
            Serial.println("DBG reboot");
            Serial.flush();
            esp_restart();
            break;
        case 7:
            prefs.begin("pf", false);
            prefs.putUChar("lmode", (uint8_t)arg);
            prefs.end();
            Serial.printf("DBG lmode=%u reboot\n", (unsigned)arg);
            Serial.flush();
            esp_restart();
            break;
        case 8:
            prefs.begin("pf", false);
            prefs.putUShort("rcount", 0);
            prefs.end();
            g_reset_count = 0;
            break; TEST_HOOK_DEBUG_CASES
        default:
            Serial.printf("DBG unknown cmd=%u\n", (unsigned)cmd);
            return;
    }
    char js[768];
    node_status_json(js, sizeof(js));
    Serial.printf("DBG cmd=%u arg=%u ok\nSTATUS %s\n", (unsigned)cmd, (unsigned)arg, js);
}

static void serial_task(void *arg) {
    (void)arg;
    esp_task_wdt_add(NULL);
    static uint8_t rx[8 + WIN2_PAYLOAD + 8];
    int rn = 0;
    for (;;) {
        esp_task_wdt_reset();
        int avail = Serial.available();
        if (avail <= 0) {
            vTaskDelay(1);
            continue;
        }
        while (avail-- > 0) {
            int c = Serial.read();
            if (c < 0) {
                break;
            }
            if (rn >= (int)sizeof(rx)) {
                rn = 0;
            }
            rx[rn++] = (uint8_t)c;
            if (rn <= 3) {
                if (rx[0] != 'P' || (rn >= 2 && rx[1] != 'F')) {
                    rn = rx[rn - 1] == 'P' ? (rx[0] = 'P', 1) : 0;
                }
                continue;
            }
            const uint8_t k = rx[2], v = rx[3];
            const bool win1 = k == 'W' && v == '1', win2 = k == 'W' && v == '2';
            const bool burn = k == 'B' && v == '1', ping = k == 'P' && v == '1', dbg = k == 'D' && v == '1';
            if (!(win1 || win2 || burn || ping || dbg)) {
                rn = 0;
                continue;
            }
            if (ping) {
                Serial.write(kMagicPing, 4);
                rn = 0;
                continue;
            }
            if (burn) {
                if (rn < 6) {
                    continue;
                }
                uint16_t n = get_u16(rx + 4);
                rn = 0;
                do_burn(n == 0 || n > 1000 ? 1 : n);
                continue;
            }
            if (dbg) {
                if (rn < 10) {
                    continue;
                }
                const uint8_t cmd = rx[4];
                const uint32_t a = get_u32(rx + 5);
                const bool ok = xor_bytes(rx + 4, 5) == rx[9];
                rn = 0;
                if (ok) {
                    handle_debug(cmd, a);
                } else {
                    Serial.println("DBG bad xor");
                }
                continue;
            }
            const int payload = win2 ? WIN2_PAYLOAD : WIN1_PAYLOAD;
            const int need = 4 + 4 + payload + 1;
            if (rn < need) {
                continue;
            }
            const uint32_t seq = get_u32(rx + 4);
            CeqtResult res;
            if (xor_bytes(rx + 4, 4 + payload) != rx[need - 1]) {
                memset(&res, 0, sizeof(res));
                res.seq = seq;
                res.status = CEQT_BAD_FRAME;
                res.done_ms = millis();
                write_result(&res);
                rn = 0;
                continue;
            }
            CeqtWindow win;
            parse_window(rx + 8, win2, &win);
            rn = 0;
            run_job(&win, seq, &res, INFER_TIMEOUT_MS);
            finish(&res);
        }
    }
}

// ---- Wi-Fi and periodic checks (Core 0) ----
static void start_web() {
    if (g_web_started || !g_link) {
        return;
    }
    const bool ok = rice_web_start(!DEPLOY_BUILD, true);
    g_web_started = true;
    Serial.printf("WEB http=%d https=%d ok=%d\n", DEPLOY_BUILD ? 0 : 1, 1, ok ? 1 : 0);
    print_heap("after_web");
}

static void sup_task(void *arg) {
    (void)arg;
    esp_task_wdt_add(NULL);
    uint32_t backoff_s = 1, next_try_ms = 0, last_sha_ms = millis();
    for (;;) {
        esp_task_wdt_reset();
        vTaskDelay(pdMS_TO_TICKS(200));
        const uint32_t now = millis();
        if (g_lmode != 1 && WIFI_SSID[0]) {
            const bool up = WiFi.status() == WL_CONNECTED;
            if (up && !g_link) {
                g_link = 1;
                if (g_down_since_ms) {
                    g_last_down_ms = now - g_down_since_ms;
                    g_down_total_ms += g_last_down_ms;
                    g_reconnects++;
                    Serial.printf("LINK up t=%u down_ms=%u ip=%s\n", (unsigned)now, (unsigned)g_last_down_ms,
                                  WiFi.localIP().toString().c_str());
                }
                backoff_s = 1;
                start_web();
            } else if (!up && g_link) {
                g_link = 0;
                g_down_since_ms = now;
                next_try_ms = now + 1000;
                Serial.printf("LINK down t=%u\n", (unsigned)now);
            } else if (!up && (int32_t)(now - next_try_ms) >= 0 TEST_HOOK_RECONNECT_ALLOWED(now)) {
                if (!g_down_since_ms) {
                    g_down_since_ms = now;
                }
                WiFi.disconnect();
                WiFi.begin(WIFI_SSID, WIFI_PASS);
                next_try_ms = now + backoff_s * 1000;
                backoff_s = backoff_s < 32 ? backoff_s * 2 : 60;
            }
        }
        if (now - last_sha_ms >= (uint32_t)INTEGRITY_PERIOD_S * 1000u) {
            last_sha_ms = now;
            g_sha_ok = ceqt_check_integrity(false);
            g_sha_checks++;
            if (!g_sha_ok) {
                g_fault = CEQT_FAULT_WEIGHTS;
                Serial.println("FAULT weights (periodic check)");
            }
        }
    }
}

// ---- boot-time latency modes (R2-11 L1/L2) ----
static void run_lmode() {
    CeqtWindow gold;
    ceqt_fill_golden_window(&gold);
    if (g_lmode == 2 && WIFI_SSID[0]) {
        WiFi.mode(WIFI_STA);
        WiFi.setSleep(false);
        WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
    CeqtResult r;
    for (int i = 0; i < 20; ++i) {
        run_job(&gold, i, &r, INFER_TIMEOUT_MS);
        Serial.printf("LRUN mode=%u i=%d us_total=%u us_enc=%u us_cross=%u us_heads=%u wifi=%d\n", (unsigned)g_lmode,
                      i, (unsigned)r.us_total, (unsigned)r.us_enc, (unsigned)r.us_cross, (unsigned)r.us_heads,
                      (int)WiFi.status());
    }
}

void setup() {
    Serial.setRxBufferSize(4096);
    Serial.begin(AUDIT_BAUD);
    delay(200);
    Serial.println("\nPico-Former rice-s3 rev2");

    g_reset_reason = (uint8_t)esp_reset_reason();
    Preferences prefs;
    prefs.begin("pf", false);
    g_reset_count = prefs.getUShort("rcount", 0);
    if (g_reset_reason != ESP_RST_POWERON && g_reset_reason != ESP_RST_SW && g_reset_reason != ESP_RST_EXT) {
        g_reset_count++;
        prefs.putUShort("rcount", g_reset_count);
    }
    g_lmode = prefs.getUChar("lmode", 0);
    prefs.end();
    Serial.printf("RESET reason=%u count=%u lmode=%u\n", (unsigned)g_reset_reason, (unsigned)g_reset_count,
                  (unsigned)g_lmode);

#ifdef TEST_EAT_SRAM
    while (heap_caps_get_free_size(MALLOC_CAP_INTERNAL) > TEST_EAT_SRAM) {
        if (!heap_caps_malloc(8192, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)) {
            break;
        }
    }
    print_heap("after_eat");
#endif

    rice_web_init();
    g_sha_ok = ceqt_check_integrity(true);
    g_sha_checks = 1;
    if (!g_sha_ok) {
        g_fault = CEQT_FAULT_WEIGHTS;
        Serial.println("FAULT weights");
    }
    print_heap("before_init");
    if (!ceqt_init()) {
        g_fault = CEQT_FAULT_MEMORY;
        Serial.println("alloc FAIL");
    }
    print_heap("after_init");
    ceqt_print_buffers();

    g_job_sem = xSemaphoreCreateBinary();
    g_done_sem = xSemaphoreCreateBinary();
    g_job_lock = xSemaphoreCreateMutex();
    if (g_fault != CEQT_FAULT_MEMORY) {
        xTaskCreatePinnedToCore(infer_task, "rice", 16384, NULL, 5, &g_infer_task, 1);
    }

    if (g_lmode == 1 || g_lmode == 2) {
        run_lmode();
    }

    if (!g_fault) {
        CeqtWindow gold;
        ceqt_fill_golden_window(&gold);
        CeqtResult first;
        run_job(&gold, 0, &first, 15000);
        ceqt_score_golden(&first);
        Serial.printf("done total=%.1f ms  dMax=%.4f  lMax=%.5f  yMax=%.4f  1h=%.1f  4h=%.1f\n",
                      first.us_total / 1000.0, first.maxabs_delta, first.maxabs_logits, first.maxabs_yhat,
                      first.pred_1h, first.pred_4h);
    } else {
        Serial.printf("FAULT state=%u: inference disabled\n", (unsigned)g_fault);
    }

    print_heap("before_wifi");
    if (g_lmode != 1 && WIFI_SSID[0]) {
        WiFi.persistent(false);
        WiFi.setAutoReconnect(false);
        if (g_lmode != 2) {
            WiFi.mode(WIFI_STA);
            WiFi.setSleep(false);
            WiFi.begin(WIFI_SSID, WIFI_PASS);
        }
        const uint32_t t0 = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - t0 < PROBE_WIFI_TIMEOUT_MS) {
            delay(100);
        }
        if (WiFi.status() == WL_CONNECTED) {
            g_link = 1;
            Serial.printf("wifi %s\n", WiFi.localIP().toString().c_str());
            start_web();
            Serial.printf("http://%s/\nhttps://%s/\n", WiFi.localIP().toString().c_str(),
                          WiFi.localIP().toString().c_str());
        } else {
            g_down_since_ms = millis();
            Serial.println("wifi skip/fail");
        }
    } else {
        Serial.println("wifi off");
    }
    print_heap("before_ready"); TEST_HOOK_SETUP

    xTaskCreatePinnedToCore(sup_task, "sup", 4096, NULL, 2, &g_sup_task, 0);
#if !DEPLOY_BUILD
    xTaskCreatePinnedToCore(serial_task, "audit", 8192, NULL, 4, &g_serial_task, 0);
#endif
    Serial.println("Pico-Former ready");
}

void loop() { vTaskDelay(pdMS_TO_TICKS(1000)); }
