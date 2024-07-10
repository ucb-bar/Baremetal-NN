#include "add1.h"


__attribute__((weak)) void NN__add1_f32(size_t n, float *z, size_t incz, float *x, size_t incx, float scalar) {
  for (size_t i = 0; i < n; i += 1) {
    z[i * incz] = x[i * incx] + scalar;
  }
}
