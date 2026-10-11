// R2-14: whole CEQT graph (encoder + cross-attention + heads) quantized by ESP-PPQ, executed by ESP-DL.
// Serial protocol on UART0 (460800 baud):
//   host -> board: "QTK1" | u32 seq | f32[96*48] tokens | u8 xor(payload)
//   board -> host: "QRS1" | u32 seq | u32 us_run | f32[48] delta_norm | f32[16] logits | u32 int_free | u32 int_min
//                  | u32 psram_free | u8 xor(payload)
// Tokens are built on the host exactly as the FP32 firmware builds them; inertia baseline, gate and thresholds
// are applied on the host, so the board result isolates the quantized graph.

#include <cstring>

#include "dl_model_base.hpp"
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

static constexpr int N_TOK = 96, D = 48, N_OUT = 48, N_EVT = 16;
static constexpr int TOK_BYTES = N_TOK * D * 4;
static constexpr uart_port_t PORT = UART_NUM_0;

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
    printf("ESP-DL CEQT full graph S3\n");
    dl::Model *model = new dl::Model(reinterpret_cast<const char *>(model_espdl), fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    auto inputs = model->get_inputs();
    auto outputs = model->get_outputs();
    dl::TensorBase *model_input = inputs.begin()->second;
    dl::TensorBase *delta_t = nullptr, *logit_t = nullptr;
    for (auto &kv : outputs) {
        if (kv.second->get_size() == N_OUT) delta_t = kv.second;
        else if (kv.second->get_size() == N_EVT) logit_t = kv.second;
    }
    if (!delta_t || !logit_t) {
        printf("output FAIL\n");
        vTaskDelay(portMAX_DELAY);
    }
    printf("input dtype=%d bytes=%d\n", (int)model_input->get_dtype(), (int)model_input->get_bytes());
    model->profile_memory();
    printf("heap_internal=%u heap_internal_min=%u heap_psram=%u\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    printf("QTK ready\n");
    fflush(stdout);

    static uint8_t frame[4 + TOK_BYTES + 1];
    static uint8_t reply[4 + 4 + 4 + 4 * (N_OUT + N_EVT) + 12 + 1];
    dl::TensorBase in_f({1, N_TOK, D}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    dl::TensorBase delta_f({1, N_OUT}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    dl::TensorBase logit_f({1, N_EVT}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    uint8_t m[4];
    while (true) {
        // resynchronise on the magic
        if (!read_exact(m, 1) || m[0] != 'Q') continue;
        if (!read_exact(m + 1, 3) || memcmp(m, "QTK1", 4) != 0) continue;
        if (!read_exact(frame, sizeof(frame))) continue;
        if (xor_of(frame, 4 + TOK_BYTES) != frame[4 + TOK_BYTES]) {
            const char nak[] = "QNAK";
            uart_write_bytes(PORT, nak, 4);
            continue;
        }
        memcpy(in_f.data, frame + 4, TOK_BYTES);
        const int64_t t0 = esp_timer_get_time();
        model_input->assign(&in_f);
        model->run();
        delta_f.assign(delta_t);
        logit_f.assign(logit_t);
        const uint32_t us = (uint32_t)(esp_timer_get_time() - t0);
        uint8_t *p = reply;
        memcpy(p, "QRS1", 4); p += 4;
        memcpy(p, frame, 4); p += 4;
        memcpy(p, &us, 4); p += 4;
        memcpy(p, delta_f.data, 4 * N_OUT); p += 4 * N_OUT;
        memcpy(p, logit_f.data, 4 * N_EVT); p += 4 * N_EVT;
        const uint32_t mem[3] = {(uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                                 (uint32_t)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
                                 (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM)};
        memcpy(p, mem, 12); p += 12;
        *p = xor_of(reply + 4, (int)(p - reply - 4)); ++p;
        uart_write_bytes(PORT, reply, p - reply);
    }
}
