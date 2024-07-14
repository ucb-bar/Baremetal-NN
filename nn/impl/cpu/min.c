#include "min.h"


__attribute__((weak)) void NN__min_f32(size_t n, float *result, const float *x, size_t incx) {
  float min = FLT_MAX;
  for (size_t i = 0; i < n; i += 1) {
    float val = x[i * incx];
    min = val < min ? val : min;
  }
  *result = min;
}

