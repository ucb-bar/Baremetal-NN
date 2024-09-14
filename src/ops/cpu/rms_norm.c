#include "ops/rms_norm.h"


__attribute__((weak)) void NN_rms_norm_f32(size_t n, float* y, size_t incy, const float* x, size_t incx, const float* w, size_t incw, float eps) {
  // calculate sum of squares
  float ss = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    ss += x[i * incx] * x[i * incx];
  }
  ss /= n;
  ss += eps;

  // normalize and scale
  // y = (x / ss) * w
  NN_mul1_f32(n, y, incy, x, incx, 1.0f / sqrtf(ss));
  NN_mul_f32(n, y, incy, y, incy, w, incw);
}

