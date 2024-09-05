#include "kernel/max.h"


__attribute__((weak)) void NN_max_i8(size_t n, int8_t *r, const int8_t *x, size_t incx) {
  int8_t max = INT8_MIN;
  for (size_t i = 0; i < n; i += 1) {
    int8_t val = x[i * incx];
    max = val > max ? val : max;
  }
  *r = max;
}

__attribute__((weak)) void NN_max_i16(size_t n, int16_t *r, const int16_t *x, size_t incx) {
  int16_t max = INT16_MIN;
  for (size_t i = 0; i < n; i += 1) {
    int16_t val = x[i * incx];
    max = val > max ? val : max;
  }
  *r = max;
}

__attribute__((weak)) void NN_max_i32(size_t n, int32_t *r, const int32_t *x, size_t incx) {
  int32_t max = INT32_MIN;
  for (size_t i = 0; i < n; i += 1) {
    int32_t val = x[i * incx];
    max = val > max ? val : max;
  }
  *r = max;
}

__attribute__((weak)) void NN_max_f16(size_t n, float16_t *r, const float16_t *x, size_t incx) {
  float16_t max = NN_float_to_half(-FLT_MAX);
  for (size_t i = 0; i < n; i += 1) {
    float16_t val = x[i * incx];
    max = NN_half_to_float(val) > NN_half_to_float(max) ? val : max;
  }
  *r = max;
}

__attribute__((weak)) void NN_max_f32(size_t n, float *r, const float *x, size_t incx) {
  float max = -FLT_MAX;
  for (size_t i = 0; i < n; i += 1) {
    float val = x[i * incx];
    max = val > max ? val : max;
  }
  *r = max;
}

