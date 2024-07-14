#include "max.h"


__attribute__((weak)) void NN__max_f32(size_t n, float *result, const float *x, size_t incx) {
  float max = -FLT_MAX;
  for (size_t i = 0; i < n; i += 1) {
    float val = x[i * incx];
    max = val > max ? val : max;
  }
  *result = max;
}

