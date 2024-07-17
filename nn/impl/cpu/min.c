#include "min.h"


__attribute__((weak)) void NN__min_i8(size_t n, int8_t *result, const int8_t *x, size_t incx) {
  int8_t min = INT8_MAX;
  for (size_t i = 0; i < n; i += 1) {
    int8_t val = x[i * incx];
    min = val < min ? val : min;
  }
  *result = min;
}

__attribute__((weak)) void NN__min_i16(size_t n, int16_t *result, const int16_t *x, size_t incx) {
  int16_t min = INT16_MAX;
  for (size_t i = 0; i < n; i += 1) {
    int16_t val = x[i * incx];
    min = val < min ? val : min;
  }
  *result = min;
}

__attribute__((weak)) void NN__min_i32(size_t n, int32_t *result, const int32_t *x, size_t incx) {
  int32_t min = INT32_MAX;
  for (size_t i = 0; i < n; i += 1) {
    int32_t val = x[i * incx];
    min = val < min ? val : min;
  }
  *result = min;
}

__attribute__((weak)) void NN__min_f16(size_t n, float16_t *result, const float16_t *x, size_t incx) {
  float16_t min = NN_float_to_half(FLT_MAX);
  for (size_t i = 0; i < n; i += 1) {
    float16_t val = x[i * incx];
    min = NN_half_to_float(val) < NN_half_to_float(min) ? val : min;
  }
  *result = min;
}

__attribute__((weak)) void NN__min_f32(size_t n, float *result, const float *x, size_t incx) {
  float min = FLT_MAX;
  for (size_t i = 0; i < n; i += 1) {
    float val = x[i * incx];
    min = val < min ? val : min;
  }
  *result = min;
}

