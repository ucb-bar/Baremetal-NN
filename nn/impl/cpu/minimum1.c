#include "minimum1.h"


__attribute__((weak)) void NN__minimum1_f16(size_t n, float16_t *y, size_t incy, float16_t *x, size_t incx, float16_t scalar) {
  for (size_t i = 0; i < n; i += 1) {
    float16_t x_val = x[i * incx];
    y[i * incy] = NN_half_to_float(x_val) < NN_half_to_float(scalar) ? x_val : scalar;
  }
}

__attribute__((weak)) void NN__minimum1_f32(size_t n, float *y, size_t incy, float *x, size_t incx, float scalar) {
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x[i * incx];
    y[i * incy] = x_val < scalar ? x_val : scalar;
  }
}

