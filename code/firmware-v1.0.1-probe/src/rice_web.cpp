#include "rice_web.h"

#include "ceqt_weights.h"

#include <WebServer.h>
#include <WiFi.h>
#include <freertos/portmacro.h>

static WebServer server(80);
static CeqtResult *g_res;
static portMUX_TYPE *g_mux;
static bool g_up;

static CeqtResult snap() {
    CeqtResult r;
    if (g_mux) {
        portENTER_CRITICAL(g_mux);
        r = *g_res;
        portEXIT_CRITICAL(g_mux);
    } else {
        r = *g_res;
    }
    return r;
}

static const char *kEvt[4] = {"hypo70", "hypo54", "hyper180", "hyper250"};
static const char *kPh[4] = {"1h", "2h", "3h", "4h"};

static int mapx(int i, int n, int x0, int x1) {
    return x0 + (x1 - x0) * i / (n - 1);
}

static int mapy(float y, int y0, int y1) {
    float t = (y - 40.f) / 210.f;
    if (t < 0) {
        t = 0;
    }
    if (t > 1) {
        t = 1;
    }
    return (int)(y1 - t * (y1 - y0));
}

static void append_poly(String &s, const float *y, int n, int x0, int x1, int y0, int y1) {
    for (int i = 0; i < n; ++i) {
        if (i) {
            s += ' ';
        }
        s += String(mapx(i, n, x0, x1));
        s += ',';
        s += String(mapy(y[i], y0, y1));
    }
}

static void handle_last() {
    CeqtResult r = snap();
    String j;
    j.reserve(480);
    j += F("{\"core\":");
    j += String(r.infer_core);
    j += F(",\"us_total\":");
    j += String(r.us_total);
    j += F(",\"g\":");
    j += String(r.g, 3);
    j += F(",\"now\":");
    j += String(r.now_mgdl, 2);
    j += F(",\"pred_1h\":");
    j += String(r.pred_1h, 2);
    j += F(",\"pred_4h\":");
    j += String(r.pred_4h, 2);
    j += F(",\"heap_free\":");
    j += String(r.heap_free);
    j += F(",\"evt\":[");
    for (int i = 0; i < CEQT_N_EVT; ++i) {
        if (i) {
            j += ',';
        }
        j += String(r.evt_on[i]);
    }
    j += F("]}");
    server.send(200, "application/json", j);
}

static void handle_root() {
    CeqtResult r = snap();
    String html;
    html.reserve(4500);
    html += F("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">");
    html += F("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
    html += F("<title>Pico-Former</title><style>");
    html += F("body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;background:#f4f1ea;color:#1c1917;}");
    html += F("main{max-width:42rem;margin:1.25rem auto;padding:0 1rem;}");
    html += F("h1{font-size:1.25rem;margin:0 0 .2rem;} .sub{color:#78716c;font-size:.85rem;margin:0 0 1rem;}");
    html += F(".nums{display:flex;gap:.75rem;margin:0 0 1rem;} .nums div{flex:1;background:#fff;border-radius:10px;padding:.7rem .8rem;box-shadow:0 1px 2px #0001;}");
    html += F(".nums b{display:block;font-size:1.45rem;} .nums span{color:#78716c;font-size:.75rem;}");
    html += F(".card{background:#fff;border-radius:10px;padding:.75rem;box-shadow:0 1px 2px #0001;margin:0 0 1rem;}");
    html += F(".grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.4rem;}");
    html += F(".chip{border-radius:8px;padding:.4rem .35rem;text-align:center;font-size:.72rem;background:#f5f5f4;color:#57534e;}");
    html += F(".chip.on{background:#fee2e2;color:#991b1b;} .chip.on.hi{background:#ffedd5;color:#9a3412;}");
    html += F("details{font-size:.8rem;color:#57534e;} svg{width:100%;height:auto;}");
    html += F("</style></head><body><main><h1>Pico-Former</h1>");
    html += F("<p class=\"sub\">板上一次前向 · 预测不是监测 · 事件分为 logit 过阈值</p><div class=\"nums\">");
    html += F("<div><span>现在</span><b>");
    html += String(r.now_mgdl, 0);
    html += F("</b></div><div><span>+1 h</span><b>");
    html += String(r.pred_1h, 0);
    html += F("</b></div><div><span>+4 h</span><b>");
    html += String(r.pred_4h, 0);
    html += F("</b></div></div><div class=\"card\"><svg viewBox=\"0 0 640 240\">");
    html += F("<rect x=\"40\" y=\"");
    html += String(mapy(180, 20, 210));
    html += F("\" width=\"580\" height=\"");
    html += String(mapy(40, 20, 210) - mapy(180, 20, 210));
    html += F("\" fill=\"#ffedd5\" opacity=\".45\"/>");
    html += F("<rect x=\"40\" y=\"");
    html += String(mapy(70, 20, 210));
    html += F("\" width=\"580\" height=\"");
    html += String(mapy(40, 20, 210) - mapy(70, 20, 210));
    html += F("\" fill=\"#fee2e2\" opacity=\".5\"/>");
    html += F("<line x1=\"388\" y1=\"20\" x2=\"388\" y2=\"210\" stroke=\"#a8a29e\" stroke-dasharray=\"4 3\"/>");
    html += F("<polyline fill=\"none\" stroke=\"#a8a29e\" stroke-width=\"1.6\" points=\"");
    append_poly(html, r.y_hist, CEQT_N_CGM, 40, 388, 20, 210);
    html += F("\"/><polyline fill=\"none\" stroke=\"#1c1917\" stroke-width=\"2\" points=\"");
    html += String(388);
    html += ',';
    html += String(mapy(r.now_mgdl, 20, 210));
    html += ' ';
    for (int i = 0; i < CEQT_N_OUT; ++i) {
        html += String(mapx(i, CEQT_N_OUT, 388, 620));
        html += ',';
        html += String(mapy(r.yhat[i], 20, 210));
        if (i + 1 < CEQT_N_OUT) {
            html += ' ';
        }
    }
    html += F("\"/></svg></div><div class=\"card\"><div class=\"grid\">");
    for (int e = 0; e < 4; ++e) {
        for (int p = 0; p < 4; ++p) {
            int i = e * 4 + p;
            html += F("<div class=\"chip");
            if (r.evt_on[i]) {
                html += (e >= 2) ? F(" on hi") : F(" on");
            }
            html += F("\">");
            html += kEvt[e];
            html += ' ';
            html += kPh[p];
            html += F("<br>");
            html += r.evt_on[i] ? F("过阈值") : F("未过");
            html += F("</div>");
        }
    }
    html += F("</div></div><details><summary>对齐与性能</summary><p>core ");
    html += String(r.infer_core);
    html += F(" · 整网 ");
    html += String(r.us_total / 1000.0f, 1);
    html += F(" ms · encode ");
    html += String(r.us_enc / 1000.0f, 1);
    html += F(" · cross ");
    html += String(r.us_cross / 1000.0f, 1);
    html += F(" · heads ");
    html += String(r.us_heads / 1000.0f, 1);
    html += F("</p><p>max-abs Δ ");
    html += String(r.maxabs_delta, 4);
    html += F(" · logit ");
    html += String(r.maxabs_logits, 5);
    html += F(" · ŷ ");
    html += String(r.maxabs_yhat, 4);
    html += F(" · g ");
    html += String(r.g, 2);
    html += F("</p><p>分数未经概率校准。推理只钉在一个核。</p></details>");
    html += F("<p class=\"sub\">subject ");
    html += String(CEQT_SUBJECT);
    html += F(" · ");
    html += WiFi.localIP().toString();
    html += F("</p></main></body></html>");
    server.send(200, "text/html; charset=utf-8", html);
}

bool rice_web_start(CeqtResult *shared, void *mux) {
    g_res = shared;
    g_mux = (portMUX_TYPE *)mux;
    if (WiFi.status() != WL_CONNECTED) {
        return false;
    }
    server.on("/", handle_root);
    server.on("/api/last", handle_last);
    server.on("/favicon.ico", []() { server.send(204); });
    server.begin();
    g_up = true;
    return true;
}

void rice_web_loop() {
    if (g_up) {
        server.handleClient();
    }
}
