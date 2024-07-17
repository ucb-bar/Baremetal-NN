#include "transpose.h"


__attribute__((weak)) void NN__transpose_i8(size_t m, size_t n, int8_t *y, const int8_t *x) {
  for (size_t i = 0; i < m; i += 1) {
    for (size_t j = 0; j < n; j += 1) {
      y[j * m + i] = x[i * n + j];
    }
  }
};

__attribute__((weak)) void NN__transpose_i16(size_t m, size_t n, int16_t *y, const int16_t *x) {
  for (size_t i = 0; i < m; i += 1) {
    for (size_t j = 0; j < n; j += 1) {
      y[j * m + i] = x[i * n + j];
    }
  }
};

__attribute__((weak)) void NN__transpose_i32(size_t m, size_t n, int32_t *y, const int32_t *x) {
  for (size_t i = 0; i < m; i += 1) {
    for (size_t j = 0; j < n; j += 1) {
      y[j * m + i] = x[i * n + j];
    }
  }
};

__attribute__((weak)) void NN__transpose_f16(size_t m, size_t n, float16_t *y, const float16_t *x) {
  for (size_t i = 0; i < m; i += 1) {
    for (size_t j = 0; j < n; j += 1) {
      y[j * m + i] = x[i * n + j];
    }
  }
};

__attribute__((weak)) void NN__transpose_f32(size_t m, size_t n, float *y, const float *x) {
  for (size_t i = 0; i < m; i += 1) {
    for (size_t j = 0; j < n; j += 1) {
      y[j * m + i] = x[i * n + j];
    }
  }
};
