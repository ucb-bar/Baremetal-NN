#ifndef __NN__SUB_H
#define __NN__SUB_H

#include <stddef.h>
#include <stdint.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__sub_U8(size_t n, uint8_t *z, uint8_t *x, uint8_t *y) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m1(n);
      vuint8m1_t vec_x = __riscv_vle8_v_u8m1(x, vl);
      vuint8m1_t vec_y = __riscv_vle8_v_u8m1(y, vl);
      vuint8m1_t vec_z = __riscv_vsub_vv_u8m1(vec_x, vec_y, vl);
      __riscv_vse8_v_u8m1(z, vec_z, vl);
      x += vl;
      y += vl;
      z += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      z[i] = x[i] - y[i];
    }
  #endif
}

static inline void NN__sub_I8(size_t n, int8_t *z, int8_t *x, int8_t *y) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m1(n);
      vint8m1_t vec_x = __riscv_vle8_v_i8m1(x, vl);
      vint8m1_t vec_y = __riscv_vle8_v_i8m1(y, vl);
      vint8m1_t vec_z = __riscv_vsub_vv_i8m1(vec_x, vec_y, vl);
      __riscv_vse8_v_i8m1(z, vec_z, vl);
      x += vl;
      y += vl;
      z += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      z[i] = x[i] - y[i];
    }
  #endif
}

static inline void NN__sub_I16(size_t n, int16_t *z, int16_t *x, int16_t *y) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vint16m1_t vec_x = __riscv_vle16_v_i16m1(x, vl);
      vint16m1_t vec_y = __riscv_vle16_v_i16m1(y, vl);
      vint16m1_t vec_z = __riscv_vsub_vv_i16m1(vec_x, vec_y, vl);
      __riscv_vse16_v_i16m1(z, vec_z, vl);
      x += vl;
      y += vl;
      z += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      z[i] = x[i] - y[i];
    }
  #endif
}

static inline void NN__sub_I32(size_t n, int32_t *z, int32_t *x, int32_t *y) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vint32m1_t vec_x = __riscv_vle32_v_i32m1(x, vl);
      vint32m1_t vec_y = __riscv_vle32_v_i32m1(y, vl);
      vint32m1_t vec_z = __riscv_vsub_vv_i32m1(vec_x, vec_y, vl);
      __riscv_vse32_v_i32m1(z, vec_z, vl);
      x += vl;
      y += vl;
      z += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      z[i] = x[i] - y[i];
    }
  #endif
}

static inline void NN__sub_F32(size_t n, float *z, float *x, float *y) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
      vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(y, vl);
      vfloat32m1_t vec_z = __riscv_vfsub_vv_f32m1(vec_x, vec_y, vl);
      __riscv_vse32_v_f32m1(z, vec_z, vl);
      x += vl;
      y += vl;
      z += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      z[i] = x[i] - y[i];
    }
  #endif
}

#endif // __NN__SUB_H
