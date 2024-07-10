#include "sub.h"


__attribute__((weak)) void NN__sub_u8(size_t n, uint8_t *z, size_t incz, uint8_t *x, size_t incx, uint8_t *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    z[i * incz] = x[i * incx] - y[i * incy];
  }
}

__attribute__((weak)) void NN__sub_i8(size_t n, int8_t *z, size_t incz, int8_t *x, size_t incx, int8_t *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    z[i * incz] = x[i * incx] - y[i * incy];
  }
}

__attribute__((weak)) void NN__sub_i16(size_t n, int16_t *z, size_t incz, int16_t *x, size_t incx, int16_t *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    z[i * incz] = x[i * incx] - y[i * incy];
  }
}

__attribute__((weak)) void NN__sub_i32(size_t n, int32_t *z, size_t incz, int32_t *x, size_t incx, int32_t *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    z[i * incz] = x[i * incx] - y[i * incy];
  }
}

__attribute__((weak)) void NN__sub_f16(size_t n, float16_t *z, size_t incz, float16_t *x, size_t incx, float16_t *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    z[i * incz] = NN_float_to_half(NN_half_to_float(x[i * incx]) - NN_half_to_float(y[i * incy]));
  }
}

__attribute__((weak)) void NN__sub_f32(size_t n, float *z, size_t incz, float *x, size_t incx, float *y, size_t incy) {
  for (size_t i = 0; i < n; i += 1) {
    z[i * incz] = x[i * incx] - y[i * incy];
  }
}

