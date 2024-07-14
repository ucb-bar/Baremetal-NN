#include "transpose.h"


__attribute__((weak)) void NN__transpose_f32(size_t m, size_t n, float *y, const float *x) {
  for (size_t i = 0; i < m; i += 1) {
    for (size_t j = 0; j < n; j += 1) {
      y[j * m + i] = x[i * n + j];
    }
  }
};
