#include "sum.h"


__attribute__((weak)) void NN__sum_u8_to_i32(size_t n, int32_t *result, uint8_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (int32_t)x[i * incx];
  }
  *result = sum;
}

__attribute__((weak)) void NN__sum_i16_to_i32(size_t n, int32_t *result, int16_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (int32_t)x[i * incx];
  }
  *result = sum;
}

__attribute__((weak)) void NN__sum_i32(size_t n, int32_t *result, int32_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx];
  }
  *result = sum;
}

__attribute__((weak)) void NN__sum_f32(size_t n, float *result, float *x, size_t incx) {
  float sum = 0.f;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx];
  }
  *result = sum;
}
