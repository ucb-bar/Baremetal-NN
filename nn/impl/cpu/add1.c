#include "impl/add1.h"



__attribute__((weak)) void NN_add1_i8(size_t n, int8_t *y, size_t incy, const int8_t *x, size_t incx, int8_t scalar) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = x[i * incx] + scalar;
  }
}

__attribute__((weak)) void NN_add1_i16(size_t n, int16_t *y, size_t incy, const int16_t *x, size_t incx, int16_t scalar) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = x[i * incx] + scalar;
  }
}

__attribute__((weak)) void NN_add1_i32(size_t n, int32_t *y, size_t incy, const int32_t *x, size_t incx, int32_t scalar) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = x[i * incx] + scalar;
  }
}

__attribute__((weak)) void NN_add1_f16(size_t n, float16_t *y, size_t incy, const float16_t *x, size_t incx, float16_t scalar) {
  for (size_t i = 0; i < n; i += incx) {
    y[i * incy] = NN_float_to_half(NN_half_to_float(x[i * incx]) + NN_half_to_float(scalar));
  }
}

__attribute__((weak)) void NN_add1_f32(size_t n, float *y, size_t incy, const float *x, size_t incx, float scalar) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = x[i * incx] + scalar;
  }
}
