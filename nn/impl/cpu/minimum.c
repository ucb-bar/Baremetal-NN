#include "minimum.h"


__attribute__((weak)) void NN__minimum_i8(size_t n, int8_t *z, size_t incz, const int8_t *x, size_t incx, const int8_t *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    int8_t x_val = x[i * incx];
    int8_t y_val = y[i * incy];
    z[i * incz] = x_val < y_val ? x_val : y_val;
  }
}

__attribute__((weak)) void NN__minimum_i16(size_t n, int16_t *z, size_t incz, const int16_t *x, size_t incx, const int16_t *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    int16_t x_val = x[i * incx];
    int16_t y_val = y[i * incy];
    z[i * incz] = x_val < y_val ? x_val : y_val;
  }
}

__attribute__((weak)) void NN__minimum_i32(size_t n, int32_t *z, size_t incz, const int32_t *x, size_t incx, const int32_t *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    int32_t x_val = x[i * incx];
    int32_t y_val = y[i * incy];
    z[i * incz] = x_val < y_val ? x_val : y_val;
  }
}

__attribute__((weak)) void NN__minimum_f16(size_t n, float16_t *z, size_t incz, const float16_t *x, size_t incx, const float16_t *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    float16_t x_val = x[i * incx];
    float16_t y_val = y[i * incy];
    z[i * incz] = NN_half_to_float(x_val) < NN_half_to_float(y_val) ? x_val : y_val;
  }
}

__attribute__((weak)) void NN__minimum_f32(size_t n, float *z, size_t incz, const float *x, size_t incx, const float *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x[i * incx];
    float y_val = y[i * incy];
    z[i * incz] = x_val < y_val ? x_val : y_val;
  }
}

