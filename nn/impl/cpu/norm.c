#include "impl/norm.h"


__attribute__((weak)) void NN_norm_f32(size_t n, float *result, const float *x, size_t incx) {
  NN_dot_f32(n, result, x, incx, x, incx);
  *result = sqrtf(*result);
}

__attribute__((weak)) void NN_norm_inv_f32(size_t n, float *result, const float *x, size_t incx) {
  NN_norm_f32(n, result, x, incx);
  *result = 1.f/(*result);
}
