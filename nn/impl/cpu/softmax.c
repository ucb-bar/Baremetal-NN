#include "softmax.h"


__attribute__((weak)) void NN__softmax_f32(size_t n, float *y, size_t incy, float *x, size_t incx) {
  // exp and sum
  float sum = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = expf(x[i * incx]);
    sum += y[i * incy];
  }
  // normalize
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] /= sum;
  }
}
