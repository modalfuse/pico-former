#pragma once
// Test-build hooks used by main.cpp. Without the test flags every hook expands to nothing, so the frozen
// firmware (16b32e5f...) is reproduced byte for byte; the hooks sit at the ends of existing lines in main.cpp
// because shifting line numbers changes the binary.
#include <stdint.h>

#ifdef TEST_LINK_DROP
#define TEST_HOOK_LINK_GLOBALS static volatile uint32_t g_link_hold_until_ms;
#define TEST_HOOK_LINK_CASES case 9: g_link_hold_until_ms = millis() + arg; WiFi.disconnect(); Serial.printf("DBG link drop hold_ms=%u\n", (unsigned)arg); break;
#define TEST_HOOK_RECONNECT_ALLOWED(now) && (int32_t)((now) - g_link_hold_until_ms) >= 0
#else
#define TEST_HOOK_LINK_GLOBALS
#define TEST_HOOK_LINK_CASES
#define TEST_HOOK_RECONNECT_ALLOWED(now)
#endif

#ifdef TEST_ENERGY
// R2-9 energy modes (src/energy_modes.cpp), driven over Wi-Fi (POST /api/energy, POST /api/window) so that the
// board needs no USB or serial connection while the PPK2 powers it; debug command 10 (arg bits 0-7 mode, bits 8-31
// parameter) runs the same modes from the audit port. g_lmode == 1 keeps the supervisor task from reconnecting
// while Wi-Fi is switched off.
typedef void (*TestInferFn)(uint32_t seq, uint32_t *us_total);
typedef uint8_t (*TestWindowFn)(const uint8_t *frame, int len, uint32_t *seq, uint32_t *us_total);
void test_energy_setup(TestInferFn infer, TestWindowFn window, uint8_t *lmode);
void test_energy(uint8_t mode, uint32_t param);
void test_energy_register(void *httpd);
#define TEST_HOOK_ENERGY_CASES case 10: test_energy((uint8_t)(arg & 0xFF), arg >> 8); break;
#define TEST_HOOK_SETUP test_energy_setup([](uint32_t s, uint32_t *us) { CeqtWindow w; ceqt_fill_golden_window(&w); CeqtResult r; run_job(&w, s, &r, INFER_TIMEOUT_MS); *us = r.us_total; }, [](const uint8_t *f, int len, uint32_t *seq, uint32_t *us) -> uint8_t { if (len != 4 + 4 + WIN2_PAYLOAD + 1 || memcmp(f, kMagicWin2, 4) != 0 || xor_bytes(f + 4, 4 + WIN2_PAYLOAD) != f[len - 1]) return CEQT_BAD_FRAME; static CeqtWindow w; static CeqtResult r; *seq = get_u32(f + 4); parse_window(f + 8, true, &w); run_job(&w, *seq, &r, INFER_TIMEOUT_MS); finish(&r); *us = r.us_total; return r.status; }, &g_lmode);
#define TEST_HOOK_WEB_REGISTER(h) test_energy_register(h);
#else
#define TEST_HOOK_ENERGY_CASES
#define TEST_HOOK_SETUP
#define TEST_HOOK_WEB_REGISTER(h)
#endif

#define TEST_HOOK_GLOBALS TEST_HOOK_LINK_GLOBALS
#define TEST_HOOK_DEBUG_CASES TEST_HOOK_LINK_CASES TEST_HOOK_ENERGY_CASES
