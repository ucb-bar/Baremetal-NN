#include "norm.h"


__attribute__((weak)) void NN__norm_f32(size_t n, float *result, const float *x, size_t incx) {
  NN__dot_f32(n, result, x, incx, x, incx);
  *result = sqrtf(*result);
}

__attribute__((weak)) void NN__norm_inv_f32(size_t n, float *result, const float *x, size_t incx) {
  NN__norm_f32(n, result, x, incx);
  *result = 1.f/(*result);
}
