#ifndef __NN__DOT_H
#define __NN__DOT_H

#include <stddef.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

#include "nn_float16.h"


static inline void NN__dot_f16(size_t n, float16_t *s, float16_t *x, float16_t *y) {
  float16_t sum = 0.0;
  
  #ifdef RVV
    size_t vlmax = __riscv_vsetvlmax_e16m1();

    vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
    vfloat16m1_t vec_s = __riscv_vfmv_v_f_f16m1(0, vlmax);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x, vl);
      vfloat16m1_t vec_y = __riscv_vle16_v_f16m1(y, vl);
      vec_s = __riscv_vfmacc_vv_f16m1(vec_s, vec_x, vec_y, vl);
        
      x += vl;
      y += vl;
      n -= vl;
    }
    vec_s = __riscv_vfredusum_vs_f16m1_f16m1(vec_s, vec_zero, vlmax);
    sum = __riscv_vfmv_f_s_f16m1_f16(vec_s);
  #else
    float sum_f32 = 0;
    for (size_t i = 0; i < n; i += 1) {
      sum_f32 += NN_halfToFloat(x[i]) * NN_halfToFloat(y[i]);
    }
    sum = NN_floatToHalf(sum_f32);
  #endif

  *s = sum;
}

static inline void NN__dot_f32(size_t n, float *s, float *x, float *y) {
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
