#include "kernel/softmax.h"


__attribute__((weak)) void NN_softmax_f16(size_t n, float16_t *y, size_t incy, const float16_t *x, size_t incx) {
  // exp and sum
  float sum = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = NN_float_to_half(expf(NN_half_to_float(x[i * incx])));
    sum += NN_half_to_float(y[i * incy]);
  }
  // normalize
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = NN_float_to_half(NN_half_to_float(y[i * incy]) / sum);
  }
}

__attribute__((weak)) void NN_softmax_f32(size_t n, float *y, size_t incy, const float *x, size_t incx) {
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
