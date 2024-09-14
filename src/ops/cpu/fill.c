#include "ops/fill.h"


__attribute__((weak)) void NN_fill_u8(size_t n, uint8_t *x, size_t incx, uint8_t s) {
  for (size_t i = 0; i < n; i += 1) {
    x[i * incx] = s;
  }
}

__attribute__((weak)) void NN_fill_i8(size_t n, int8_t *x, size_t incx, int8_t s) {
  for (size_t i = 0; i < n; i += 1) {
    x[i * incx] = s;
  }
}

__attribute__((weak)) void NN_fill_i16(size_t n, int16_t *x, size_t incx, int16_t s) {
  for (size_t i = 0; i < n; i += 1) {
    x[i * incx] = s;
  }
}

__attribute__((weak)) void NN_fill_i32(size_t n, int32_t *x, size_t incx, int32_t s) {
  for (size_t i = 0; i < n; i += 1) {
    x[i * incx] = s;
  }
}

__attribute__((weak)) void NN_fill_f16(size_t n, float16_t *x, size_t incx, float16_t s) {
  for (size_t i = 0; i < n; i += 1) {
    x[i * incx] = s;
  }
}

__attribute__((weak)) void NN_fill_f32(size_t n, float *x, size_t incx, float s) {
  for (size_t i = 0; i < n; i += 1) {
    x[i * incx] = s;
  }
}
