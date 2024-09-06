#include "ops/mul.h"


__attribute__((weak)) void NN_mul_i8(size_t n, int8_t *y, size_t incy, const int8_t *x1, size_t incx1, const int8_t *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = x1[i * incx1] * x2[i * incx2];
  }
}

__attribute__((weak)) void NN_mul_i16(size_t n, int16_t *y, size_t incy, const int16_t *x1, size_t incx1, const int16_t *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = x1[i * incx1] * x2[i * incx2];
  }
}

__attribute__((weak)) void NN_mul_i32(size_t n, int32_t *y, size_t incy, const int32_t *x1, size_t incx1, const int32_t *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = x1[i * incx1] * x2[i * incx2];
  }
}

__attribute__((weak)) void NN_mul_f16(size_t n, float16_t *y, size_t incy, const float16_t *x1, size_t incx1, const float16_t *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = NN_float_to_half(NN_half_to_float(x1[i * incx1]) * NN_half_to_float(x2[i * incx2]));
  }
}

__attribute__((weak)) void NN_mul_f32(size_t n, float *y, size_t incy, const float *x1, size_t incx1, const float *x2, size_t incx2) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = x1[i * incx1] * x2[i * incx2];
  }
}
