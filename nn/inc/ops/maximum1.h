#ifndef __NN__MAXIMUM1_H
#define __NN__MAXIMUM1_H

#include <stddef.h>
#include <stdint.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__maximum1_F32(size_t n, float *y, float *x, float v) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
      vfloat32m1_t vec_v = __riscv_vfmv_v_f_f32m1(v, vl);
      vfloat32m1_t vec_y = __riscv_vfmax_vv_f32m1(vec_x, vec_v, vl);
      __riscv_vse32_v_f32m1(y, vec_y, vl);
      x += vl;
      y += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      float x_val = x[i];
      y[i] = x_val > v ? x_val : v;
    }
  #endif
}

#endif // __NN__MAXIMUM1_H
