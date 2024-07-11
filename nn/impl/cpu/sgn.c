#include "sgn.h"


__attribute__((weak)) void NN__sgn_f32(size_t n, float *y, size_t incy, float *x, size_t incx) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = (x[i * incx] > 0.f) ? 1.f : ((x[i * incx] < 0.f) ? -1.f : 0.f);
  }
}

