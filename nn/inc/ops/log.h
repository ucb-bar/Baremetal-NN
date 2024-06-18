#ifndef __NN__LOG_H
#define __NN__LOG_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__log_F32(size_t n, float *y, float *x) {
  for (size_t i = 0; i < n; i += 1) {
    y[i] = logf(x[i]);
  }
}


#endif // __NN__LOG_H
