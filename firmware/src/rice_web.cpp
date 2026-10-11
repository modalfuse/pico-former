#include "rice_web.h"

#include "api_token.gen.h"
#include "https_cert.gen.h"

#include <Arduino.h>
#include <esp_http_server.h>
#include <esp_https_server.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <stdio.h>
#include <string.h>

#ifndef HTTPS_MAX_SESSIONS
#define HTTPS_MAX_SESSIONS 2
#endif
#ifndef TLS_PSRAM
#define TLS_PSRAM 1
#endif
#if TLS_PSRAM
#include <esp_heap_caps.h>
#include <mbedtls/platform.h>
#endif
#ifndef CEQT_GPIO_HTTP
#define CEQT_GPIO_HTTP 5
#endif

enum { RING_N = 64, LAST_N = 1024 };

struct HistItem {
    uint32_t seq, t0_s, done_ms;
    uint16_t evt_bits;
    uint8_t status, flags;
    float pred_1h, pred_4h;
};

static SemaphoreHandle_t g_lock;
static SemaphoreHandle_t g_out_lock;  // guards the static response buffers shared by both server tasks
static char g_last_evt[LAST_N];    // tail of /api/last with event bits
static char g_last_noevt[LAST_N];  // same, events replaced by null (stale or no prediction)
static uint32_t g_last_done_ms;
static uint8_t g_last_status = 0xFF;
static HistItem g_ring[RING_N];
static uint32_t g_ring_n;
static volatile uint32_t g_stale_ms = 600000;
static httpd_handle_t g_http, g_https;
static NetStats g_st;

static const char kPage[] =
    "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Pico-Former</title>"
    "<style>body{font-family:sans-serif;max-width:40rem;margin:1rem auto;padding:0 1rem;background:#f4f1ea}"
    ".n{display:flex;gap:.6rem}.n div{flex:1;background:#fff;border-radius:8px;padding:.6rem}"
    ".g{display:grid;grid-template-columns:repeat(4,1fr);gap:.3rem;margin-top:.8rem}"
    ".c{background:#fff;border-radius:6px;padding:.3rem;text-align:center;font-size:.75rem}.on{background:#fee2e2}"
    "#s{margin:.6rem 0;color:#57534e}</style></head><body><h1>Pico-Former</h1><p id=\"s\">-</p>"
    "<div class=\"n\"><div>现在<b id=\"a\"></b></div><div>+1 h<b id=\"b\"></b></div><div>+4 h<b id=\"c\"></b></div></div>"
    "<div class=\"g\" id=\"e\"></div><script>"
    "const T=location.hash.slice(1),E=['hypo70','hypo54','hyper180','hyper250'];"
    "const R={10:'历史缺测或越界',11:'数据过期',12:'输入无效',20:'输出无效',21:'权重校验失败',22:'推理失败',23:'内存不足'};"
    "async function f(){try{const r=await fetch('/api/last',{headers:T?{Authorization:'Bearer '+T}:{}});"
    "const j=await r.json();const s=document.getElementById('s');"
    "s.textContent=j.stale?'过期（'+j.age_s+' s 前）':(j.status?'无预测：'+(R[j.status]||j.status):'序号 '+j.seq+'，'+j.age_s+' s 前');"
    "a.textContent=' '+j.now;b.textContent=' '+j.pred_1h;c.textContent=' '+j.pred_4h;let h='';"
    "for(let i=0;i<16;i++){const v=j.evt?j.evt[i]:null;h+='<div class=\"c'+(v?' on':'')+'\">'+E[i>>2]+' '+(1+(i&3))+' h<br>'+(v===null?'—':(v?'过阈值':'未过'))+'</div>'}"
    "e.innerHTML=h}catch(x){document.getElementById('s').textContent='连接失败'}}f();setInterval(f,5000);"
    "</script></body></html>";

static bool is_https(httpd_req_t *req) { return g_https && req->handle == g_https; }

static void count(httpd_req_t *req, esp_err_t err, uint32_t t0) {
    const uint32_t dt = micros() - t0;
    const uint32_t hw = uxTaskGetStackHighWaterMark(NULL);
    xSemaphoreTake(g_lock, portMAX_DELAY);
    if (is_https(req)) {
        g_st.https_req++;
        g_st.https_fail += err != ESP_OK;
        g_st.https_stack_hw = hw;
    } else {
        g_st.http_req++;
        g_st.http_fail += err != ESP_OK;
        g_st.http_stack_hw = hw;
    }
    if (dt > g_st.handler_us_max) {
        g_st.handler_us_max = dt;
    }
    xSemaphoreGive(g_lock);
    digitalWrite(CEQT_GPIO_HTTP, LOW);
}

static bool authorized(httpd_req_t *req) {
    if (API_TOKEN[0] == '\0') {
        return true;
    }
    char hdr[96];
    if (httpd_req_get_hdr_value_str(req, "Authorization", hdr, sizeof(hdr)) == ESP_OK &&
        strncmp(hdr, "Bearer ", 7) == 0 && strcmp(hdr + 7, API_TOKEN) == 0) {
        return true;
    }
    httpd_resp_set_status(req, "401 Unauthorized");
    httpd_resp_send(req, "{\"error\":\"unauthorized\"}", HTTPD_RESP_USE_STRLEN);
    return false;
}

static esp_err_t h_root(httpd_req_t *req) {
    const uint32_t t0 = micros();
    digitalWrite(CEQT_GPIO_HTTP, HIGH);
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    esp_err_t err = httpd_resp_send(req, kPage, sizeof(kPage) - 1);
    count(req, err, t0);
    return ESP_OK;
}

static esp_err_t h_last(httpd_req_t *req) {
    const uint32_t t0 = micros();
    digitalWrite(CEQT_GPIO_HTTP, HIGH);
    if (!authorized(req)) {
        count(req, ESP_OK, t0);
        return ESP_OK;
    }
    static char out[LAST_N + 64];
    xSemaphoreTake(g_out_lock, portMAX_DELAY);
    xSemaphoreTake(g_lock, portMAX_DELAY);
    const uint32_t age_ms = g_last_status == 0xFF ? 0 : millis() - g_last_done_ms;
    const bool stale = g_last_status != 0xFF && age_ms > g_stale_ms;
    const char *tail = (stale || g_last_status != CEQT_OK) ? g_last_noevt : g_last_evt;
    int n = snprintf(out, sizeof(out), "{\"age_s\":%.1f,\"stale\":%d,%s", age_ms / 1000.f, stale ? 1 : 0,
                     g_last_status == 0xFF ? "\"seq\":0,\"status\":255,\"evt\":null}" : tail);
    xSemaphoreGive(g_lock);
    httpd_resp_set_type(req, "application/json");
    esp_err_t err = httpd_resp_send(req, out, n);
    xSemaphoreGive(g_out_lock);
    count(req, err, t0);
    return ESP_OK;
}

static esp_err_t h_history(httpd_req_t *req) {
    const uint32_t t0 = micros();
    digitalWrite(CEQT_GPIO_HTTP, HIGH);
    if (!authorized(req)) {
        count(req, ESP_OK, t0);
        return ESP_OK;
    }
    uint32_t since = 0;
    char q[48], v[16];
    if (httpd_req_get_url_query_str(req, q, sizeof(q)) == ESP_OK &&
        httpd_query_key_value(q, "since", v, sizeof(v)) == ESP_OK) {
        since = strtoul(v, nullptr, 10);
    }
    static HistItem snap[RING_N];
    xSemaphoreTake(g_out_lock, portMAX_DELAY);
    xSemaphoreTake(g_lock, portMAX_DELAY);
    const uint32_t total = g_ring_n;
    memcpy(snap, g_ring, sizeof(snap));
    const uint32_t now = millis();
    xSemaphoreGive(g_lock);
    httpd_resp_set_type(req, "application/json");
    esp_err_t err = httpd_resp_send_chunk(req, "[", 1);
    const uint32_t first = total > RING_N ? total - RING_N : 0;
    bool comma = false;
    char row[160];
    for (uint32_t k = first; k < total && err == ESP_OK; ++k) {
        const HistItem &h = snap[k % RING_N];
        if (h.seq <= since) {
            continue;
        }
        int n = snprintf(row, sizeof(row),
                         "%s{\"seq\":%u,\"t0\":%u,\"age_s\":%.1f,\"status\":%u,\"flags\":%u,\"evt_bits\":%u,"
                         "\"pred_1h\":%.2f,\"pred_4h\":%.2f}",
                         comma ? "," : "", (unsigned)h.seq, (unsigned)h.t0_s, (now - h.done_ms) / 1000.f,
                         (unsigned)h.status, (unsigned)h.flags, (unsigned)h.evt_bits, h.pred_1h, h.pred_4h);
        err = httpd_resp_send_chunk(req, row, n);
        comma = true;
    }
    if (err == ESP_OK) {
        err = httpd_resp_send_chunk(req, "]", 1);
    }
    if (err == ESP_OK) {
        err = httpd_resp_send_chunk(req, nullptr, 0);
    }
    xSemaphoreGive(g_out_lock);
    count(req, err, t0);
    return ESP_OK;
}

static esp_err_t h_status(httpd_req_t *req) {
    const uint32_t t0 = micros();
    digitalWrite(CEQT_GPIO_HTTP, HIGH);
    if (!authorized(req)) {
        count(req, ESP_OK, t0);
        return ESP_OK;
    }
    char buf[768];
    size_t n = node_status_json(buf, sizeof(buf));
    httpd_resp_set_type(req, "application/json");
    esp_err_t err = httpd_resp_send(req, buf, n);
    count(req, err, t0);
    return ESP_OK;
}

static void register_uris(httpd_handle_t h) {
    static const httpd_uri_t uris[] = {
        {"/", HTTP_GET, h_root, nullptr},
        {"/api/last", HTTP_GET, h_last, nullptr},
        {"/api/history", HTTP_GET, h_history, nullptr},
        {"/api/status", HTTP_GET, h_status, nullptr},
    };
    for (const auto &u : uris) {
        httpd_register_uri_handler(h, &u);
    } TEST_HOOK_WEB_REGISTER(h)
}

#if TLS_PSRAM
// The prebuilt sdkconfig forces mbedTLS allocations into internal RAM (CONFIG_MBEDTLS_INTERNAL_MEM_ALLOC); the
// 16 KB record buffers then no longer fit next to the inference buffers. Large blocks go to PSRAM instead.
static void *tls_calloc(size_t n, size_t size) {
    const size_t bytes = n * size;
    void *p = NULL;
    if (bytes >= 4096) {
        p = heap_caps_calloc(n, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    if (!p) {
        p = heap_caps_calloc(n, size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    return p;
}
static void tls_free(void *p) { heap_caps_free(p); }
#endif

void rice_web_init() {
    if (!g_lock) {
#if TLS_PSRAM
        mbedtls_platform_set_calloc_free(tls_calloc, tls_free);
#endif
        g_lock = xSemaphoreCreateMutex();
        g_out_lock = xSemaphoreCreateMutex();
        pinMode(CEQT_GPIO_HTTP, OUTPUT);
        digitalWrite(CEQT_GPIO_HTTP, LOW);
    }
}

bool rice_web_start(bool http, bool https) {
    rice_web_init();
    bool ok = true;
    if (http && !g_http) {
        httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
        cfg.core_id = 0;
        cfg.task_priority = 5;
        cfg.stack_size = 6144;
        cfg.max_open_sockets = 7;
        cfg.lru_purge_enable = true;
        cfg.server_port = 80;
        cfg.ctrl_port = 32768;
        ok = httpd_start(&g_http, &cfg) == ESP_OK;
        if (ok) {
            register_uris(g_http);
            g_st.http_up = 1;
        }
    }
    if (https && !g_https) {
        httpd_ssl_config_t cfg = HTTPD_SSL_CONFIG_DEFAULT();
        cfg.httpd.core_id = 0;
        cfg.httpd.task_priority = 5;
        cfg.httpd.max_open_sockets = HTTPS_MAX_SESSIONS;
        cfg.httpd.lru_purge_enable = true;
        cfg.httpd.ctrl_port = 32769;
        cfg.cacert_pem = (const uint8_t *)HTTPS_CERT_PEM;
        cfg.cacert_len = sizeof(HTTPS_CERT_PEM);
        cfg.prvtkey_pem = (const uint8_t *)HTTPS_KEY_PEM;
        cfg.prvtkey_len = sizeof(HTTPS_KEY_PEM);
        const bool ok_s = httpd_ssl_start(&g_https, &cfg) == ESP_OK;
        if (ok_s) {
            register_uris(g_https);
            g_st.https_up = 1;
        }
        ok = ok && ok_s;
    }
    return ok;
}

void rice_web_publish(const CeqtResult *r) {
    char evt[64], *p = evt;
    uint16_t bits = 0;
    *p++ = '[';
    for (int i = 0; i < CEQT_N_EVT; ++i) {
        *p++ = r->evt_on[i] ? '1' : '0';
        *p++ = i + 1 < CEQT_N_EVT ? ',' : ']';
        bits |= (uint16_t)(r->evt_on[i] ? 1 : 0) << i;
    }
    *p = 0;
    char body[LAST_N - 32];
    snprintf(body, sizeof(body),
             "\"seq\":%u,\"status\":%u,\"flags\":%u,\"t0\":%u,\"core\":%d,\"us_total\":%u,\"g\":%.3f,\"now\":%.1f,"
             "\"pred_1h\":%.1f,\"pred_4h\":%.1f,\"int_free\":%u",
             (unsigned)r->seq, (unsigned)r->status, (unsigned)r->flags, (unsigned)r->t0_s, r->infer_core,
             (unsigned)r->us_total, r->g, r->now_mgdl, r->pred_1h, r->pred_4h, (unsigned)r->heap_free);
    xSemaphoreTake(g_lock, portMAX_DELAY);
    snprintf(g_last_evt, sizeof(g_last_evt), "%s,\"evt\":%s}", body, evt);
    snprintf(g_last_noevt, sizeof(g_last_noevt), "%s,\"evt\":null}", body);
    g_last_done_ms = r->done_ms;
    g_last_status = r->status;
    HistItem &h = g_ring[g_ring_n % RING_N];
    h = {r->seq, r->t0_s, r->done_ms, r->status == CEQT_OK ? bits : (uint16_t)0, r->status, r->flags, r->pred_1h,
         r->pred_4h};
    g_ring_n++;
    xSemaphoreGive(g_lock);
}

void rice_web_stats(NetStats *out) {
    if (!g_lock) {
        memset(out, 0, sizeof(*out));
        return;
    }
    xSemaphoreTake(g_lock, portMAX_DELAY);
    *out = g_st;
    xSemaphoreGive(g_lock);
}

void rice_web_set_stale_ms(uint32_t ms) { g_stale_ms = ms; }
uint32_t rice_web_stale_ms() { return g_stale_ms; }
