#include "kernel/sqrt.h"


__attribute__((weak)) void NN_sqrt_f16(size_t n, float16_t *y, size_t incy, const float16_t *x, size_t incx) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = NN_float_to_half(sqrtf(NN_half_to_float(x[i * incx])));
  }
}

__attribute__((weak)) void NN_sqrt_f32(size_t n, float *y, size_t incy, const float *x, size_t incx) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = sqrtf(x[i * incx]);
  }
}

