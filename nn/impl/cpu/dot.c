#include "dot.h"


__attribute__((weak)) void NN__dot_i8_to_i32(size_t n, int32_t *result, const int8_t *x, size_t incx, const int8_t *y, size_t incy) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx] * y[i * incy];
  }
  *result = sum;
}

__attribute__((weak)) void NN__dot_i16_to_i32(size_t n, int32_t *result, const int16_t *x, size_t incx, const int16_t *y, size_t incy) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx] * y[i * incy];
  }
  *result = sum;
}

__attribute__((weak)) void NN__dot_i32(size_t n, int32_t *result, const int32_t *x, size_t incx, const int32_t *y, size_t incy) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx] * y[i * incy];
  }
  *result = sum;
}

__attribute__((weak)) void NN__dot_f16(size_t n, float16_t *result, const float16_t *x, size_t incx, const float16_t *y, size_t incy) {
  float sum_f32 = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum_f32 += NN_half_to_float(x[i * incx]) * NN_half_to_float(y[i * incy]);
  }
  *result = NN_float_to_half(sum_f32);
}

__attribute__((weak)) void NN__dot_f32(size_t n, float *result, const float *x, size_t incx, const float *y, size_t incy) {
  float sum = 0.0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i * incx] * y[i * incy];
  }
  *result = sum;
}
