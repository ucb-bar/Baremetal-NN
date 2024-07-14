#include "sqrt.h"


__attribute__((weak)) void NN__sqrt_f32(size_t n, float *y, size_t incy, const float *x, size_t incx) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = sqrtf(x[i * incx]);
  }
}

