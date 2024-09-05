#include "kernel/min.h"


__attribute__((weak)) void NN_min_i8(size_t n, int8_t *r, const int8_t *x, size_t incx) {
  int8_t min = INT8_MAX;
  for (size_t i = 0; i < n; i += 1) {
    int8_t val = x[i * incx];
    min = val < min ? val : min;
  }
  *r = min;
}

__attribute__((weak)) void NN_min_i16(size_t n, int16_t *r, const int16_t *x, size_t incx) {
  int16_t min = INT16_MAX;
  for (size_t i = 0; i < n; i += 1) {
    int16_t val = x[i * incx];
    min = val < min ? val : min;
  }
  *r = min;
}

__attribute__((weak)) void NN_min_i32(size_t n, int32_t *r, const int32_t *x, size_t incx) {
  int32_t min = INT32_MAX;
  for (size_t i = 0; i < n; i += 1) {
    int32_t val = x[i * incx];
    min = val < min ? val : min;
  }
  *r = min;
}

__attribute__((weak)) void NN_min_f16(size_t n, float16_t *r, const float16_t *x, size_t incx) {
  float16_t min = NN_float_to_half(FLT_MAX);
  for (size_t i = 0; i < n; i += 1) {
    float16_t val = x[i * incx];
    min = NN_half_to_float(val) < NN_half_to_float(min) ? val : min;
  }
  *r = min;
}

__attribute__((weak)) void NN_min_f32(size_t n, float *r, const float *x, size_t incx) {
  float min = FLT_MAX;
  for (size_t i = 0; i < n; i += 1) {
    float val = x[i * incx];
    min = val < min ? val : min;
  }
  *r = min;
}

