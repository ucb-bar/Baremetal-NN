#ifndef __NN__SGN_H
#define __NN__SGN_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__sgn_f32(size_t n, float *y, float *x) {
  for (size_t i = 0; i < n; i += 1) {
    y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f);
  }
}


#endif // __NN__SGN_H
