#include "kernel/norm.h"


__attribute__((weak)) void NN_norm_f32(size_t n, float *r, const float *x, size_t incx) {
  NN_dot_f32(n, r, x, incx, x, incx);
  *r = sqrtf(*r);
}

__attribute__((weak)) void NN_norm_inv_f32(size_t n, float *r, const float *x, size_t incx) {
  NN_norm_f32(n, r, x, incx);
  *r = 1.f/(*r);
}
