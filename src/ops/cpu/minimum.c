#include "ops/minimum.h"


__attribute__((weak)) void NN_minimum_i8(size_t n, int8_t *y, size_t incy, const int8_t *x1, size_t incx1, const int8_t *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    int8_t x1_val = x1[i * incx1];
    int8_t x2_val = x2[i * incx2];
    y[i * incy] = x1_val < x2_val ? x1_val : x2_val;
  }
}

__attribute__((weak)) void NN_minimum_i16(size_t n, int16_t *y, size_t incy, const int16_t *x1, size_t incx1, const int16_t *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    int16_t x1_val = x1[i * incx1];
    int16_t x2_val = x2[i * incx2];
    y[i * incy] = x1_val < x2_val ? x1_val : x2_val;
  }
}

__attribute__((weak)) void NN_minimum_i32(size_t n, int32_t *y, size_t incy, const int32_t *x1, size_t incx1, const int32_t *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    int32_t x1_val = x1[i * incx1];
    int32_t x2_val = x2[i * incx2];
    y[i * incy] = x1_val < x2_val ? x1_val : x2_val;
  }
}

__attribute__((weak)) void NN_minimum_f16(size_t n, float16_t *y, size_t incy, const float16_t *x1, size_t incx1, const float16_t *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    float16_t x1_val = x1[i * incx1];
    float16_t x2_val = x2[i * incx2];
    y[i * incy] = NN_half_to_float(x1_val) < NN_half_to_float(x2_val) ? x1_val : x2_val;
  }
}

__attribute__((weak)) void NN_minimum_f32(size_t n, float *y, size_t incy, const float *x1, size_t incx1, const float *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    float x1_val = x1[i * incx1];
    float x2_val = x2[i * incx2];
    y[i * incy] = x1_val < x2_val ? x1_val : x2_val;
  }
}

