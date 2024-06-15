#ifndef __NN__SUM_H
#define __NN__SUM_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__sum_F32(size_t n, float *s, float *x) {
  float sum = 0.0;
  for (int i = 0; i < n; i += 1) {
    sum += (float)x[i];
  }
  *s = sum;
}



#endif // __NN__SUM_H
