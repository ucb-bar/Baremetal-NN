#ifndef __NN__MIN_H
#define __NN__MIN_H

#include <stddef.h>
#include <stdint.h>
#include <float.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__min_f32(size_t n, float *s, float *x) {
  float min = FLT_MAX;
  
  #ifdef RVV
    vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(min, 1);
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
      vec_min = __riscv_vfredmin_vs_f32m1_f32m1(vec_x, vec_min, vl);
      x += vl;
      n -= vl;
    }
    min = __riscv_vfmv_f_s_f32m1_f32(vec_min);
  #else
    for (size_t i = 0; i < n; i += 1) {
      float val = x[i];
      min = val < min ? val : min;
    }
  #endif

  *s = min;
}

#endif // __NN__MIN_H
