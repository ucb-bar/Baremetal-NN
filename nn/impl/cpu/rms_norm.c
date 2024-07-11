#include "rms_norm.h"


__attribute__((weak)) void NN__rms_norm_f32(size_t n, float* y, size_t incy, float* x, size_t incx, float* w, size_t incw) {
  // calculate sum of squares
  float ss = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    ss += x[i * incx] * x[i * incx];
  }
  ss /= n;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = w[i * incw] * (ss * x[i * incx]);
  }
}

