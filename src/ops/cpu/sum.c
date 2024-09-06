#include "ops/sum.h"


__attribute__((weak)) void NN_sum_u8_to_i32(size_t n, int32_t *r, const uint8_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (int32_t)x[i * incx];
  }
  *r = sum;
}

__attribute__((weak)) void NN_sum_i8_to_i32(size_t n, int32_t *r, const int8_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (int32_t)x[i * incx];
  }
  *r = sum;
}

__attribute__((weak)) void NN_sum_i16_to_i32(size_t n, int32_t *r, const int16_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (int32_t)x[i * incx];
  }
  *r = sum;
}

__attribute__((weak)) void NN_sum_i32(size_t n, int32_t *r, const int32_t *x, size_t incx) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx];
  }
  *r = sum;
}

__attribute__((weak)) void NN_sum_f16(size_t n, float16_t *r, const float16_t *x, size_t incx) {
  float sum = 0.f;
  for (size_t i = 0; i < n; i += 1) {
    sum += NN_half_to_float(x[i * incx]);
  }
  *r = NN_float_to_half(sum);
}

__attribute__((weak)) void NN_sum_f32(size_t n, float *r, const float *x, size_t incx) {
  float sum = 0.f;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx];
  }
  *r = sum;
}
