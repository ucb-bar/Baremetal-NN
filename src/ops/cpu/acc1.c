#include "ops/acc1.h"


__attribute__((weak)) void NN_acc1_i8(size_t n, int8_t *r, size_t incr, int8_t s) {
  for (size_t i = 0; i < n; i += 1) {
    r[i + incr] += s;
  }
}

__attribute__((weak)) void NN_acc1_i16(size_t n, int16_t *r, size_t incr, int16_t s) {
  for (size_t i = 0; i < n; i += 1) {
    r[i + incr] += s;
  }
}

__attribute__((weak)) void NN_acc1_i32(size_t n, int32_t *r, size_t incr, int32_t s) {
  for (size_t i = 0; i < n; i += 1) {
    r[i + incr] += s;
  }
}

__attribute__((weak)) void NN_acc1_f16(size_t n, float16_t *r, size_t incr, float16_t s) {
  for (size_t i = 0; i < n; i += 1) {
    r[i + incr] = NN_float_to_half(NN_half_to_float(r[i * incr]) + NN_half_to_float(s));
  }
}

__attribute__((weak)) void NN_acc1_f32(size_t n, float *r, size_t incr, float s) {
  for (size_t i = 0; i < n; i += 1) {
    r[i + incr] += s;
  }
}