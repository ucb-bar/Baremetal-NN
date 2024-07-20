#include "impl/sum.h"


__attribute__((weak)) void NN__sum_u8_to_i32(size_t n, int32_t *result, const uint8_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (int32_t)x[i * incx];
  }
  *result = sum;
}

__attribute__((weak)) void NN__sum_i8_to_i32(size_t n, int32_t *result, const int8_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (int32_t)x[i * incx];
  }
  *result = sum;
}

__attribute__((weak)) void NN__sum_i16_to_i32(size_t n, int32_t *result, const int16_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (int32_t)x[i * incx];
  }
  *result = sum;
}

__attribute__((weak)) void NN__sum_i32(size_t n, int32_t *result, const int32_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx];
  }
  *result = sum;
}

__attribute__((weak)) void NN__sum_f16(size_t n, float16_t *result, const float16_t *x, size_t incx) {
  float sum = 0.f;
  for (size_t i = 0; i < n; i += 1) {
    sum += NN_half_to_float(x[i * incx]);
  }
  *result = NN_float_to_half(sum);
}

__attribute__((weak)) void NN__sum_f32(size_t n, float *result, const float *x, size_t incx) {
  float sum = 0.f;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx];
  }
  *result = sum;
}
