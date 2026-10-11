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

// Result status. Any status other than CEQT_OK carries no event bits.
enum CeqtStatus : uint8_t {
    CEQT_OK = 0,
    CEQT_BUSY = 1,
    CEQT_BAD_FRAME = 2,
    CEQT_NO_PRED_GAP = 10,      // R1/R3: CGM more than CEQT_CGM_TOL outside 40-400 mg/dL (treated as missing)
    CEQT_NO_PRED_STALE = 11,    // R4: last CGM older than 15 min at t0
    CEQT_NO_PRED_INVALID = 12,  // R8: non-finite input
    CEQT_FAULT_OUTPUT = 20,     // M1
    CEQT_FAULT_WEIGHTS = 21,    // M2
    CEQT_FAULT_INFER = 22,      // M3
    CEQT_FAULT_MEMORY = 23,     // M4
};

enum {
    CEQT_FLAG_MARKS_TRUNCATED = 1 << 0,  // R7, set by the sender
    CEQT_FLAG_MARKS_DROPPED = 1 << 1,    // R5/R6: invalid marks removed on the node
    CEQT_FLAG_STALE = 1 << 2,            // M7: result older than the staleness limit
    CEQT_FLAG_CGM_CLAMPED = 1 << 3,      // R1: sensor-limit reading clamped to 40 or 400 mg/dL
};

// Reportable range of Dexcom G6/G7 and Medtronic Enlite/Guardian 3 sensors; readings beyond it are LOW/HIGH.
// Values within CEQT_CGM_TOL outside the range (unit-conversion rounding of LOW/HIGH) are clamped to the limit.
#define CEQT_CGM_MIN 40.f
#define CEQT_CGM_MAX 400.f
#define CEQT_CGM_TOL 2.f
#define CEQT_STALE_INPUT_S 900u
#define CEQT_YHAT_MAX 600.f
// log1p of the per-type dose limits: meal 250 g, bolus/correction 30 U (Omnipod 5 maximum bolus),
// basal 35 U/h (highest programmable basal rate among MiniMed 780G, Omnipod 5 and t:slim X2).
#define CEQT_LOG1P_MEAL_MAX 5.525453f
#define CEQT_LOG1P_BOLUS_MAX 3.433987f
#define CEQT_LOG1P_BASAL_MAX 3.583519f

struct CeqtWindow {
    float y_hist[CEQT_N_CGM];
    float marks[CEQT_N_MARK * 3];
    float mask[CEQT_N_MARK];
    uint16_t n_marks;
    uint8_t quality_ok;
    uint8_t flags;
    uint32_t t0_s;
    uint32_t t_last_s;
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
    uint8_t status;
    uint8_t flags;
    uint8_t n_dropped;
    uint32_t t0_s;
    uint32_t done_ms;
};

bool ceqt_init();
bool ceqt_ready();
void ceqt_fill_golden_window(CeqtWindow *win);
void ceqt_infer_window(const CeqtWindow *win, CeqtResult *out);
void ceqt_score_golden(CeqtResult *out);
void ceqt_infer_golden(CeqtResult *out);
const float *ceqt_thresholds();
void ceqt_print_buffers();
bool ceqt_check_integrity(bool verbose);
void ceqt_debug_poison_next();
