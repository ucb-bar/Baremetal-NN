#ifndef __NN__ADD_H
#define __NN__ADD_H

#include <stddef.h>
#include <stdint.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__add_I8(size_t n, int8_t *z, int8_t *x, int8_t *y) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m1(n);
      vint8m1_t vec_x = __riscv_vle8_v_i8m1(x, vl);
      vint8m1_t vec_y = __riscv_vle8_v_i8m1(y, vl);
      vint8m1_t vec_z = __riscv_vadd_vv_i8m1(vec_x, vec_y, vl);
      __riscv_vse8_v_i8m1(z, vec_z, vl);
      x += vl;
      y += vl;
      z += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      z[i] = x[i] + y[i];
    }
  #endif
}

static inline void NN__add_F32(size_t n, float *z, float *x, float *y) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
      vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(y, vl);
      vfloat32m1_t vec_z = __riscv_vfadd_vv_f32m1(vec_x, vec_y, vl);
      __riscv_vse32_v_f32m1(z, vec_z, vl);
      x += vl;
      y += vl;
      z += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      z[i] = x[i] + y[i];
    }
  #endif
}

#endif // __NN__ADD_H
