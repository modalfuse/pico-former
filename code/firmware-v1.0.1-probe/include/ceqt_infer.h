#pragma once

#include <stdint.h>

enum {
    CEQT_N_CGM = 72,
    CEQT_N_MARK = 24,
    CEQT_N_TOK = 96,
    CEQT_D = 48,
    CEQT_H = 4,
    CEQT_DK = 12,
    CEQT_N_OUT = 48,
    CEQT_N_EVT = 16,
    CEQT_FF = 96,
};

struct CeqtWindow {
    float y_hist[CEQT_N_CGM];
    float marks[CEQT_N_MARK * 3];
    float mask[CEQT_N_MARK];
    uint16_t n_marks;
    uint8_t quality_ok;
};

struct CeqtResult {
    float y_hist[CEQT_N_CGM];
    float yhat[CEQT_N_OUT];
    float logits[CEQT_N_EVT];
    float delta[CEQT_N_OUT];
    float y_iner[CEQT_N_OUT];
    float g;
    float now_mgdl;
    float pred_1h;
    float pred_4h;
    float maxabs_delta;
    float maxabs_logits;
    float maxabs_yhat;
    uint32_t us_total;
    uint32_t us_enc;
    uint32_t us_cross;
    uint32_t us_heads;
    uint32_t heap_free;
    int infer_core;
    int evt_on[CEQT_N_EVT];
    uint32_t seq;
};

bool ceqt_init();
void ceqt_fill_golden_window(CeqtWindow *win);
void ceqt_infer_window(const CeqtWindow *win, CeqtResult *out);
void ceqt_score_golden(CeqtResult *out);
void ceqt_infer_golden(CeqtResult *out);
const float *ceqt_thresholds();
void ceqt_probe_print();
void probe_heap(const char *stage);
