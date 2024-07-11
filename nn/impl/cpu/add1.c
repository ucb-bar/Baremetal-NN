#include "add1.h"


__attribute__((weak)) void NN__add1_f32(size_t n, float *y, size_t incy, float *x, size_t incx, float scalar) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = x[i * incx] + scalar;
  }
}
