#pragma once

#include "ceqt_infer.h"

#include <stddef.h>
#include <stdint.h>

struct NetStats {
    uint32_t http_req, http_fail, https_req, https_fail;
    uint32_t handler_us_max;
    uint32_t http_stack_hw, https_stack_hw;
    uint8_t http_up, https_up;
};

void rice_web_init();
// Starts the plain HTTP server (port 80) and/or the HTTPS server (port 443); both run their own task on Core 0.
bool rice_web_start(bool http, bool https);
// Called after every completed result (serial task, Core 0): pre-renders /api/last and appends to the history ring.
void rice_web_publish(const CeqtResult *r);
void rice_web_stats(NetStats *out);
void rice_web_set_stale_ms(uint32_t ms);
uint32_t rice_web_stale_ms();

// Provided by main.cpp: node status as a JSON object.
size_t node_status_json(char *buf, size_t n);

#include "test_hooks.h"
