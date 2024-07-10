#include "acc1.h"


__attribute__((weak)) void NN__acc1_f32(size_t n, float *result, size_t incx, float scalar) {
  for (size_t i = 0; i < n; i += incx) {
    result[i] += scalar;
  }
}