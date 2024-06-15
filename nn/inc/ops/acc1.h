#ifndef __NN__ACC1_H
#define __NN__ACC1_H

#include <stddef.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__acc1_F32(size_t n, float *y, float v) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(y, vl);
      vfloat32m1_t vec_v = __riscv_vfmv_v_f_f32m1(v, vl);
      vec_y = __riscv_vfadd_vv_f32m1(vec_y, vec_v, vl);
      __riscv_vse32_v_f32m1(y, vec_y, vl);
      y += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      y[i] += v;
    }
  #endif
}

#endif // __NN__ADD1_H
