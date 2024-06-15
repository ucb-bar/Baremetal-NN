#ifndef __NN__NORM_H
#define __NN__NORM_H

#include <stddef.h>
#include <math.h>
#ifdef RVV
  #include <riscv_vector.h>
#endif

#include "dot.h"

static inline void NN__norm_F32(size_t n, float *s, float *x) {
  NN__dot_F32(n, s, x, x);
  *s = sqrtf(*s);
}

static inline void NN__norm_inv_F32(size_t n, float *s, float *x) {
  NN__norm_F32(n, s, x);
  *s = 1.f/(*s);
}


#endif // __NN__NORM_H
