#include "maximum1.h"


__attribute__((weak)) void NN__maximum1_f32(size_t n, float *y, size_t incy, float *x, size_t incx, float scalar) {
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x[i * incx];
    y[i * incy] = x_val > scalar ? x_val : scalar;
  }
}

