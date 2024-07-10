#ifndef __NN__SUM_H
#define __NN__SUM_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__sum_i16_to_i32(size_t n, uint32_t *s, int16_t *x) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (int32_t)x[i];
  }
  *s = sum;
}

static inline void NN__sum_i32(size_t n, uint32_t *s, int32_t *x) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x[i];
  }
  *s = sum;
}

static inline void NN__sum_f32(size_t n, float *s, float *x) {
  float sum = 0.0;
  for (size_t i = 0; i < n; i += 1) {
    sum += (float)x[i];
  }
  *s = sum;
}



#endif // __NN__SUM_H
