#ifndef __NN__DOT_H
#define __NN__DOT_H

#include <stddef.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__dot_F32(size_t n, float *s, float *x, float *y) {
  float sum = 0.0;
  
  #ifdef RVV
    size_t vlmax = __riscv_vsetvlmax_e32m1();

    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
      vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(y, vl);
      vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_x, vec_y, vl);
        
      x += vl;
      y += vl;
      n -= vl;
    }
    vec_s = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_s);
  #else
    for (size_t i = 0; i < n; i += 1) {
      sum += x[i] * y[i];
    }
  #endif

  *s = sum;
}

#endif // __NN__DOT_H
