// R2-9 energy-measurement modes (compiled only in build env s3_energy, see build_src_filter).
// During the PPK2 measurement the board is powered only by the PPK2 on the 3V3 rail (no USB-C, no DC jack, no
// serial adapter), so the modes are driven over Wi-Fi:
//   POST /api/energy?mode=M&param=P   start a mode (returns at once; the mode runs in its own task)
//   GET  /api/energy                  {"busy","id","done","log"}: log holds the ENERGY lines of the last command
//   POST /api/window                  one PFW2 frame (as on the audit port); inferred and published to /api/last
// Debug command 10 on the audit port runs the same modes (arg bits 0-7 mode, bits 8-31 parameter).
// Modes follow the PPK2 measurement plan, section 4:
//   0  restore: Wi-Fi connected, modem sleep off, CPU 240 MHz, cycle marker low
//   1  E1  Wi-Fi off, param golden-window inferences back to back (default 80)
//   2  E2  Wi-Fi connected, modem sleep off, idle param s (default 60)
//   3  E3  Wi-Fi connected, modem sleep off, param inferences (default 80)
//   4  E4  Wi-Fi connected, modem sleep on, idle param s (default 60)
//   5  E5  Wi-Fi connected, modem sleep off, idle param s while the host polls /api/last (default 60)
//   6  E6  Wi-Fi off, cycles of one inference + light sleep; param bits 0-7 cycles (default 10),
//          bits 8-23 sleep seconds (default 10)
//   7  E7  Wi-Fi connected, modem sleep on; the host then posts one window per 5 min cycle
//   8/9    cycle marker (GPIO 6) high / low
//   10     CPU frequency param MHz (80, 160 or 240)
//   11     release the UART0 pins (GPIO 43/44 to inputs with pull-down) so that the unpowered on-board CH340
//          is not fed through its RXD pin; serial output stops until the next reset
//   12  E7 with windows stored in flash (energy_windows.gen.h): modem sleep on; param bits 0-7 cycles
//          (default 12), bits 8-23 period s (default 300); each cycle infers the next stored window and publishes it
//          to /api/last and /api/history, so no data reaches the node over the air during the run
// The cycle marker is high for the whole idle interval in E2, E4 and E5, for each wake period in E6 and from
// window reception (or read from flash in mode 12) to result publication in E7. Started over Wi-Fi, E1 and E6 reconnect when they finish.
#ifdef TEST_ENERGY

#include "api_token.gen.h"
#include "ceqt_infer.h"
#include "energy_windows.gen.h"
#include "test_hooks.h"

#include <Arduino.h>
#include <WiFi.h>
#include <esp_heap_caps.h>
#include <esp_http_server.h>
#include <esp_sleep.h>
#include <esp_task_wdt.h>
#include <stdarg.h>

#include "wifi_secrets.gen.h"

#ifndef TEST_GPIO_CYCLE
#define TEST_GPIO_CYCLE 6
#endif

enum { LOG_N = 2048, FRAME_N = 768 };

static TestInferFn g_infer;
static TestWindowFn g_window;
static uint8_t *g_lmode;
static TaskHandle_t g_task;
static volatile bool g_busy, g_serial_on = true, g_via_http;
static volatile uint32_t g_id, g_done;
static uint8_t g_mode;
static uint32_t g_param;
static bool g_is_window;
// Buffers live in PSRAM: internal RAM left after the inference buffers is what Wi-Fi and lwIP allocate from, and
// this build must not leave less of it than the frozen firmware does.
static uint8_t *g_frame;
static int g_frame_len;
static char *g_log, *g_out;
static int g_log_n;

static void log_line(const char *fmt, ...) {
    char line[192];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n > 0 && g_log_n + n + 1 < LOG_N) {
        memcpy(g_log + g_log_n, line, n);
        g_log_n += n;
        g_log[g_log_n++] = ';';
        g_log[g_log_n] = '\0';
    }
    if (g_serial_on) {
        Serial.printf("ENERGY %s\n", line);
    }
}

static void wdt() {
    if (xTaskGetCurrentTaskHandle() != g_task) {
        esp_task_wdt_reset();
    }
}

static void mark(bool on) { digitalWrite(TEST_GPIO_CYCLE, on ? HIGH : LOW); }

static void wait_s(uint32_t s) {
    const uint32_t t0 = millis();
    while (millis() - t0 < s * 1000u) {
        wdt();
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

static void wifi_off() {
    *g_lmode = 1;
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    vTaskDelay(pdMS_TO_TICKS(500));
}

static bool wifi_on(bool modem_sleep) {
    if (WiFi.status() != WL_CONNECTED) {
        WiFi.mode(WIFI_STA);
        WiFi.begin(WIFI_SSID, WIFI_PASS);
        const uint32_t t0 = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
            wdt();
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }
    const bool up = WiFi.status() == WL_CONNECTED;
    WiFi.setSleep(modem_sleep);
    if (up) {
        *g_lmode = 0;
    }
    return up;
}

static void infer_n(uint32_t n) {
    uint32_t mn = 0xFFFFFFFF, mx = 0, us = 0;
    uint64_t sum = 0;
    const uint32_t t0 = millis();
    for (uint32_t i = 0; i < n; ++i) {
        g_infer(i, &us);
        sum += us;
        mn = us < mn ? us : mn;
        mx = us > mx ? us : mx;
        wdt();
    }
    log_line("infer n=%u wall_ms=%u min_us=%u max_us=%u mean_us=%u", (unsigned)n, (unsigned)(millis() - t0),
             (unsigned)mn, (unsigned)mx, (unsigned)(n ? sum / n : 0));
}

static void report(uint8_t mode, uint32_t param) {
    const bool up = WiFi.status() == WL_CONNECTED;
    log_line("mode=%u param=%u cpu_mhz=%u wifi=%d sleep=%d rssi=%d ch=%d int_free=%u int_min=%u", (unsigned)mode,
             (unsigned)param, (unsigned)getCpuFrequencyMhz(), (int)WiFi.status(), WiFi.getSleep() ? 1 : 0,
             up ? (int)WiFi.RSSI() : 0, up ? (int)WiFi.channel() : 0,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
}

static void run_mode(uint8_t mode, uint32_t param) {
    switch (mode) {
        case 0:
            mark(false);
            setCpuFrequencyMhz(240);
            wifi_on(false);
            break;
        case 1:
        case 3:
            if (mode == 1) {
                wifi_off();
            } else {
                wifi_on(false);
            }
            report(mode, param);
            infer_n(param ? param : 80);
            break;
        case 2:
        case 4:
        case 5:
            wifi_on(mode == 4);
            report(mode, param);
            mark(true);
            wait_s(param ? param : 60);
            mark(false);
            break;
        case 6: {
            const uint32_t cycles = (param & 0xFF) ? (param & 0xFF) : 10;
            const uint32_t sleep_s = (param >> 8) ? (param >> 8) : 10;
            wifi_off();
            report(mode, param);
            uint32_t us = 0, slept_ms = 0;
            for (uint32_t i = 0; i < cycles; ++i) {
                mark(true);
                g_infer(i, &us);
                log_line("e6 i=%u us=%u slept_ms=%u", (unsigned)i, (unsigned)us, (unsigned)slept_ms);
                if (g_serial_on) {
                    Serial.flush();
                }
                mark(false);
                esp_sleep_enable_timer_wakeup((uint64_t)sleep_s * 1000000ull);
                const int64_t t0 = esp_timer_get_time();
                esp_light_sleep_start();
                slept_ms = (uint32_t)((esp_timer_get_time() - t0) / 1000);
                wdt();
            }
            log_line("e6 last_slept_ms=%u", (unsigned)slept_ms);
            break;
        }
        case 7:
            wifi_on(true);
            break;
        case 12: {
            const uint32_t cycles = (param & 0xFF) ? (param & 0xFF) : 12;
            const uint32_t period_ms = ((param >> 8) ? (param >> 8) : 300) * 1000u;
            wifi_on(true);
            report(mode, param);
            for (uint32_t i = 0; i < cycles; ++i) {
                const uint32_t t0 = millis();
                uint32_t seq = 0, us = 0;
                mark(true);
                const uint8_t st = g_window(ENERGY_WINDOWS[i % ENERGY_N_WINDOWS], ENERGY_FRAME_LEN, &seq, &us);
                mark(false);
                log_line("r i=%u seq=%u st=%u us=%u", (unsigned)i, (unsigned)seq, (unsigned)st, (unsigned)us);
                while (i + 1 < cycles && millis() - t0 < period_ms) {
                    wdt();
                    vTaskDelay(pdMS_TO_TICKS(100));
                }
            }
            break;
        }
        case 8:
        case 9:
            mark(mode == 8);
            break;
        case 10:
            setCpuFrequencyMhz(param == 80 || param == 160 ? param : 240);
            break;
        case 11:
            if (g_serial_on) {
                Serial.println("ENERGY uart0 pins released");
                Serial.flush();
            }
            g_serial_on = false;
            pinMode(43, INPUT_PULLDOWN);
            pinMode(44, INPUT_PULLDOWN);
            break;
        default:
            log_line("unknown mode=%u", (unsigned)mode);
            return;
    }
    if (g_via_http && (mode == 1 || mode == 6)) {
        wifi_on(false);
    }
    report(mode, param);
}

static void energy_task(void *arg) {
    (void)arg;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        g_log_n = 0;
        g_log[0] = '\0';
        if (g_is_window) {
            uint32_t seq = 0, us = 0;
            mark(true);
            const uint8_t st = g_window(g_frame, g_frame_len, &seq, &us);
            mark(false);
            log_line("window seq=%u status=%u us=%u", (unsigned)seq, (unsigned)st, (unsigned)us);
        } else {
            vTaskDelay(pdMS_TO_TICKS(300));  // lets the HTTP response leave before Wi-Fi may be switched off
            g_via_http = true;
            run_mode(g_mode, g_param);
            g_via_http = false;
        }
        g_done = g_id;
        g_busy = false;
    }
}

void test_energy_setup(TestInferFn infer, TestWindowFn window, uint8_t *lmode) {
    g_infer = infer;
    g_window = window;
    g_lmode = lmode;
    pinMode(TEST_GPIO_CYCLE, OUTPUT);
    mark(false);
    g_frame = (uint8_t *)heap_caps_malloc(FRAME_N, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    g_log = (char *)heap_caps_calloc(1, LOG_N, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    g_out = (char *)heap_caps_malloc(LOG_N + 96, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    xTaskCreatePinnedToCore(energy_task, "energy", 4096, NULL, 3, &g_task, 0);
}

void test_energy(uint8_t mode, uint32_t param) {
    if (g_busy) {
        Serial.println("ENERGY busy");
        return;
    }
    g_busy = true;
    g_log_n = 0;
    g_log[0] = '\0';
    run_mode(mode, param);
    g_busy = false;
}

// ---- HTTP interface ----
static bool authorized(httpd_req_t *req) {
    char hdr[96];
    if (API_TOKEN[0] == '\0' ||
        (httpd_req_get_hdr_value_str(req, "Authorization", hdr, sizeof(hdr)) == ESP_OK &&
         strncmp(hdr, "Bearer ", 7) == 0 && strcmp(hdr + 7, API_TOKEN) == 0)) {
        return true;
    }
    httpd_resp_set_status(req, "401 Unauthorized");
    httpd_resp_send(req, "{\"error\":\"unauthorized\"}", HTTPD_RESP_USE_STRLEN);
    return false;
}

static esp_err_t reply(httpd_req_t *req, const char *status, const char *body) {
    httpd_resp_set_status(req, status);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, body, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t start(httpd_req_t *req) {
    char out[48];
    g_id = g_id + 1;
    snprintf(out, sizeof(out), "{\"accepted\":1,\"id\":%u}", (unsigned)g_id);
    xTaskNotifyGive(g_task);
    return reply(req, "202 Accepted", out);
}

static esp_err_t h_energy_post(httpd_req_t *req) {
    if (!authorized(req)) {
        return ESP_OK;
    }
    char q[64], v[16];
    if (!g_task || httpd_req_get_url_query_str(req, q, sizeof(q)) != ESP_OK ||
        httpd_query_key_value(q, "mode", v, sizeof(v)) != ESP_OK) {
        return reply(req, "400 Bad Request", "{\"error\":\"mode\"}");
    }
    if (g_busy) {
        return reply(req, "409 Conflict", "{\"error\":\"busy\"}");
    }
    g_busy = true;
    g_is_window = false;
    g_mode = (uint8_t)strtoul(v, nullptr, 10);
    g_param = httpd_query_key_value(q, "param", v, sizeof(v)) == ESP_OK ? strtoul(v, nullptr, 10) : 0;
    return start(req);
}

static esp_err_t h_window_post(httpd_req_t *req) {
    if (!authorized(req)) {
        return ESP_OK;
    }
    if (!g_task || req->content_len == 0 || req->content_len > FRAME_N) {
        return reply(req, "400 Bad Request", "{\"error\":\"length\"}");
    }
    if (g_busy) {
        return reply(req, "409 Conflict", "{\"error\":\"busy\"}");
    }
    g_busy = true;
    int got = 0;
    while (got < (int)req->content_len) {
        const int n = httpd_req_recv(req, (char *)g_frame + got, req->content_len - got);
        if (n <= 0) {
            g_busy = false;
            return reply(req, "400 Bad Request", "{\"error\":\"recv\"}");
        }
        got += n;
    }
    g_frame_len = got;
    g_is_window = true;
    return start(req);
}

static esp_err_t h_energy_get(httpd_req_t *req) {
    if (!authorized(req)) {
        return ESP_OK;
    }
    if (!g_task) {
        return reply(req, "503 Service Unavailable", "{\"error\":\"starting\"}");
    }
    snprintf(g_out, LOG_N + 96, "{\"busy\":%d,\"id\":%u,\"done\":%u,\"log\":\"%s\"}", g_busy ? 1 : 0,
             (unsigned)g_id, (unsigned)g_done, g_busy ? "" : g_log);
    return reply(req, "200 OK", g_out);
}

void test_energy_register(void *httpd) {
    static const httpd_uri_t uris[] = {
        {"/api/energy", HTTP_POST, h_energy_post, nullptr},
        {"/api/energy", HTTP_GET, h_energy_get, nullptr},
        {"/api/window", HTTP_POST, h_window_post, nullptr},
    };
    for (const auto &u : uris) {
        httpd_register_uri_handler((httpd_handle_t)httpd, &u);
    }
}

#endif
