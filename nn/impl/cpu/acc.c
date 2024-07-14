#include "acc.h"


__attribute__((weak)) void NN__acc_i8(size_t n, int8_t *y, size_t incy, const int8_t *x, size_t incx) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] += x[i * incx];
  }
}

__attribute__((weak)) void NN__acc_f32(size_t n, float *y, size_t incy, const float *x, size_t incx) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] += x[i * incx];
  }
}
