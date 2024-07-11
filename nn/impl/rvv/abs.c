#include <riscv_vector.h>
#include "abs.h"

#ifdef RVV

void NN__abs_i8(size_t n, int8_t *y, size_t incy, int8_t *x, size_t incx) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vint8m1_t vec_x = __riscv_vlse8_v_i8m1(x, sizeof(int8_t) * incx, vl);
    vint8m1_t vec_neg_x = __riscv_vneg_v_i8m1(vec_x, vl);
    vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(vec_x, 0, vl);
    vint8m1_t vec_abs_x = __riscv_vmerge_vvm_i8m1(vec_x, vec_neg_x, mask, vl);
    __riscv_vsse8_v_i8m1(y, sizeof(int8_t) * incy, vec_abs_x, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}

void NN__abs_i16(size_t n, int16_t *y, size_t incy, int16_t *x, size_t incx) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e16m1(n);
    vint16m1_t vec_x = __riscv_vlse16_v_i16m1(x, sizeof(int16_t) * incx, vl);
    vint16m1_t vec_neg_x = __riscv_vneg_v_i16m1(vec_x, vl);
    vbool16_t mask = __riscv_vmslt_vx_i16m1_b16(vec_x, 0, vl);
    vint16m1_t vec_abs_x = __riscv_vmerge_vvm_i16m1(vec_x, vec_neg_x, mask, vl);
    __riscv_vsse16_v_i16m1(y, sizeof(int16_t) * incy, vec_abs_x, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}

void NN__abs_i32(size_t n, int32_t *y, size_t incy, int32_t *x, size_t incx) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vint32m1_t vec_x = __riscv_vlse32_v_i32m1(x, sizeof(int32_t) * incx, vl);
    vint32m1_t vec_neg_x = __riscv_vneg_v_i32m1(vec_x, vl);
    vbool32_t mask = __riscv_vmslt_vx_i32m1_b32(vec_x, 0, vl);
    vint32m1_t vec_abs_x = __riscv_vmerge_vvm_i32m1(vec_x, vec_neg_x, mask, vl);
    __riscv_vsse32_v_i32m1(y, sizeof(int32_t) * incy, vec_abs_x, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}

// void NN__abs_f16(size_t n, float16_t *y, size_t incy, float16_t *x, size_t incx) {
//   while (n > 0) {
//     size_t vl = __riscv_vsetvl_e16m1(n);
//     vfloat16m1_t vec_x = __riscv_vlse16_v_f16m1(x, sizeof(float16_t) * incx, vl);
//     vfloat16m1_t vec_y = __riscv_vfabs_v_f16m1(vec_x, vl);
//     __riscv_vse16_v_f16m1(y, sizeof(float16_t) * incy, vec_y, vl);
//     x += vl;
//     y += vl;
//     n -= vl;
//   }
// }

void NN__abs_f32(size_t n, float *y, size_t incy, float *x, size_t incx) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, sizeof(float) * incx, vl);
    vfloat32m1_t vec_y = __riscv_vfabs_v_f32m1(vec_x, vl);
    __riscv_vsse32_v_f32m1(y, sizeof(float) * incy, vec_y, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}

#endif