#ifndef __NN__ACC_H
#define __NN__ACC_H

#include <stddef.h>
#include <stdint.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__acc_i8(size_t n, int8_t *y, int8_t *x) {
  for (size_t i = 0; i < n; i += 1) {
    y[i] += x[i];
  }
}

static inline void NN__acc_f32(size_t n, float *y, float *x) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
      vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(y, vl);
      vec_y = __riscv_vfadd_vv_f32m1(vec_y, vec_x, vl);
      __riscv_vse32_v_f32m1(y, vec_y, vl);
      x += vl;
      y += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      y[i] += x[i];
    }
  #endif
}

#endif // __NN__ACC_H
