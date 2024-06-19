#ifndef __NN__SOFTMAX_H
#define __NN__SOFTMAX_H

#include <stddef.h>
#include <math.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__softmax_F32(size_t n, float *y, float *x, size_t stride) {
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < n * stride; i += stride) {
    y[i] = expf(x[i]);
    sum += y[i];
  }
  // normalize
  for (int i = 0; i < n * stride; i += stride) {
    y[i] /= sum;
  }
}


#endif // __NN__SOFTMAX_H
