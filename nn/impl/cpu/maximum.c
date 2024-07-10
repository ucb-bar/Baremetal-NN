#include "maximum.h"


__attribute__((weak)) void NN__maximum_f32(size_t n, float *z, size_t incz, float *x, size_t incx, float *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x[i * incx];
    float y_val = y[i * incy];
    z[i * incz] = x_val > y_val ? x_val : y_val;
  }
}

