#ifndef __NN__TRANSPOSE_H
#define __NN__TRANSPOSE_H

#include <stddef.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__transpose_f32(size_t n, size_t m, float *y, float *x) {
  #ifdef RVV
    for (size_t i = 0; i < m; i += 1) {
      size_t k = n;
      while (k > 0) {
        size_t vl = __riscv_vsetvl_e32m1(k);
        vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
        __riscv_vsse32_v_f32m1(y, sizeof(float) * m, vec_x, vl);
        x += vl;
        y += vl * m;
        k -= vl;
      }
    }
  #else
    for (size_t i = 0; i < m; i += 1) {
      for (size_t j = 0; j < n; j += 1) {
        y[j * m + i] = x[i * n + j];
      }
    }
  #endif
};


#endif // __NN__TRANSPOSE_H
