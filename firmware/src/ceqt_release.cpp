#include <esp_heap_caps.h>

void ceqt_release_partial(float *a, float *b, float *c, float *d, float *e, float *f, float *g) {
    float *bufs[] = {a, b, c, d, e, f, g};
    for (float *p : bufs) {
        heap_caps_free(p);
    }
}
