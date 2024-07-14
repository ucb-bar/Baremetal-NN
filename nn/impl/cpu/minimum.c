#include "minimum.h"


__attribute__((weak)) void NN__minimum_f32(size_t n, float *z, size_t incz, const float *x, size_t incx, const float *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x[i * incx];
    float y_val = y[i * incy];
    z[i * incz] = x_val < y_val ? x_val : y_val;
  }
}

