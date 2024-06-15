#ifndef __NN__TRANSPOSE_H
#define __NN__TRANSPOSE_H

#include <stddef.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__transpose_F32(size_t n, size_t m, float *y, float *x) {
  #ifdef RVV
    for (size_t i = 0; i < m; i += 1) {
      size_t k = n;
      while (k > 0) {
        vl = __riscv_vsetvl_e32(k);
        vfloat32_t vec_x = __riscv_vle32_v_f32(x, vl);
        __riscv_vsse32(y, sizeof(float) * m, vec_x, vl);
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
