#include <riscv_vector.h>
#include "sub.h"

#ifdef RVV


void NN__sub_u8(size_t n, uint8_t *z, size_t incz, uint8_t *x, size_t incx, uint8_t *y, size_t incy) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vuint8m1_t vec_x = __riscv_vlse8_v_u8m1(x, incx, vl);
    vuint8m1_t vec_y = __riscv_vlse8_v_u8m1(y, incy, vl);
    vuint8m1_t vec_z = __riscv_vsub_vv_u8m1(vec_x, vec_y, vl);
    __riscv_vsse8_v_u8m1(z, incz, vec_z, vl);
    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}

void NN__sub_i8(size_t n, int8_t *z, size_t incz, int8_t *x, size_t incx, int8_t *y, size_t incy) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vint8m1_t vec_x = __riscv_vlse8_v_i8m1(x, incx, vl);
    vint8m1_t vec_y = __riscv_vlse8_v_i8m1(y, incy, vl);
    vint8m1_t vec_z = __riscv_vsub_vv_i8m1(vec_x, vec_y, vl);
    __riscv_vsse8_v_i8m1(z, incz, vec_z, vl);
    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}

void NN__sub_i16(size_t n, int16_t *z, size_t incz, int16_t *x, size_t incx, int16_t *y, size_t incy) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e16m1(n);
    vint16m1_t vec_x = __riscv_vlse16_v_i16m1(x, incx, vl);
    vint16m1_t vec_y = __riscv_vlse16_v_i16m1(y, incy, vl);
    vint16m1_t vec_z = __riscv_vsub_vv_i16m1(vec_x, vec_y, vl);
    __riscv_vsse16_v_i16m1(z, incz, vec_z, vl);
    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}

void NN__sub_i32(size_t n, int32_t *z, size_t incz, int32_t *x, size_t incx, int32_t *y, size_t incy) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vint32m1_t vec_x = __riscv_vlse32_v_i32m1(x, incx, vl);
    vint32m1_t vec_y = __riscv_vlse32_v_i32m1(y, incy, vl);
    vint32m1_t vec_z = __riscv_vsub_vv_i32m1(vec_x, vec_y, vl);
    __riscv_vsse32_v_i32m1(z, incz, vec_z, vl);
    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}

void NN__sub_f32(size_t n, float *z, size_t incz, float *x, size_t incx, float *y, size_t incy) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, incx, vl);
    vfloat32m1_t vec_y = __riscv_vlse32_v_f32m1(y, incy, vl);
    vfloat32m1_t vec_z = __riscv_vfsub_vv_f32m1(vec_x, vec_y, vl);
    __riscv_vsse32_v_f32m1(z, incz, vec_z, vl);
    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}

#endif
