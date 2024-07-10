#ifndef __NN__SQR_H
#define __NN__SQR_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__sqr_f32(size_t n, float *y, float *x) {
  for (size_t i = 0; i < n; i += 1) {
    y[i] = x[i] * x[i];
  }
}


#endif // __NN__SQR_H
