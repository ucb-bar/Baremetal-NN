#include "rms_norm.h"

__attribute__((weak)) void NN__rms_norm_f32(size_t n, float* y, size_t incy, float* x, size_t incx, float* w, size_t incw, float eps) {

  // TODO: for some reason, passing eps as float is not working, the function is getting random values
  // so we are passing it as a pointer and dereferencing it here temporarily

  // calculate sum of squares
  float ss = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    ss += x[i * incx] * x[i * incx];
  }
  ss /= n;
  ss += eps;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = w[i * incw] * (ss * x[i * incx]);
  }
}

