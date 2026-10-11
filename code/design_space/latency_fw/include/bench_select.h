#pragma once
#if defined(CFG_d32_l1)
#include "bench_w_d32_l1.h"
#elif defined(CFG_d32_l2)
#include "bench_w_d32_l2.h"
#elif defined(CFG_d32_l3)
#include "bench_w_d32_l3.h"
#elif defined(CFG_d48_l1)
#include "bench_w_d48_l1.h"
#elif defined(CFG_base)
#include "bench_w_base.h"
#elif defined(CFG_d48_l3)
#include "bench_w_d48_l3.h"
#elif defined(CFG_d64_l1)
#include "bench_w_d64_l1.h"
#elif defined(CFG_d64_l2)
#include "bench_w_d64_l2.h"
#elif defined(CFG_d64_l3)
#include "bench_w_d64_l3.h"
#elif defined(CFG_d96_l2)
#include "bench_w_d96_l2.h"
#elif defined(CFG_L48)
#include "bench_w_L48.h"
#elif defined(CFG_L60)
#include "bench_w_L60.h"
#elif defined(CFG_L84)
#include "bench_w_L84.h"
#elif defined(CFG_h2)
#include "bench_w_h2.h"
#elif defined(CFG_h8)
#include "bench_w_h8.h"
#else
#error "define CFG_<config>"
#endif
