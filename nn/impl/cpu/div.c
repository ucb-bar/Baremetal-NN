#include "div.h"


__attribute__((weak)) void NN__div_f32(size_t n, float *z, size_t incz, const float *x, size_t incx, const float *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    z[i * incz] = x[i * incx] / y[i * incy];
  }
}
