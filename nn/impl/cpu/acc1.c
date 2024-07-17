#include "acc1.h"


__attribute__((weak)) void NN__acc1_i8(size_t n, int8_t *result, size_t incr, int8_t scalar) {
  for (size_t i = 0; i < n; i += 1) {
    result[i + incr] += scalar;
  }
}

__attribute__((weak)) void NN__acc1_i16(size_t n, int16_t *result, size_t incr, int16_t scalar) {
  for (size_t i = 0; i < n; i += 1) {
    result[i + incr] += scalar;
  }
}

__attribute__((weak)) void NN__acc1_i32(size_t n, int32_t *result, size_t incr, int32_t scalar) {
  for (size_t i = 0; i < n; i += 1) {
    result[i + incr] += scalar;
  }
}

__attribute__((weak)) void NN__acc1_f16(size_t n, float16_t *result, size_t incr, float16_t scalar) {
  for (size_t i = 0; i < n; i += 1) {
    result[i + incr] = NN_float_to_half(NN_half_to_float(result[i * incr]) + NN_half_to_float(scalar));
  }
}

__attribute__((weak)) void NN__acc1_f32(size_t n, float *result, size_t incr, float scalar) {
  for (size_t i = 0; i < n; i += 1) {
    result[i + incr] += scalar;
  }
}