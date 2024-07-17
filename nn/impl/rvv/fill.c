#include <riscv_vector.h>
#include "fill.h"

#ifdef RVV

void NN__fill_i8(size_t n, int8_t *x, size_t incx, int8_t scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vint8m1_t vec_x = __riscv_vmv_v_x_i8m1(scalar, vl);
    __riscv_vsse8_v_i8m1(x, sizeof(int8_t) * incx, vec_x, vl);
    x += vl;
    n -= vl;
  }
}

void NN__fill_i16(size_t n, int16_t *x, size_t incx, int16_t scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e16m1(n);
    vint16m1_t vec_x = __riscv_vmv_v_x_i16m1(scalar, vl);
    __riscv_vsse16_v_i16m1(x, sizeof(int16_t) * incx, vec_x, vl);
    x += vl;
    n -= vl;
  }
}

void NN__fill_i32(size_t n, int32_t *x, size_t incx, int32_t scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vint32m1_t vec_x = __riscv_vmv_v_x_i32m1(scalar, vl);
    __riscv_vsse32_v_i32m1(x, sizeof(int32_t) * incx, vec_x, vl);
    x += vl;
    n -= vl;
  }
}

// void NN__fill_f16(size_t n, float16_t *x, size_t incx, float16_t scalar) {
//   while (n > 0) {
//     size_t vl = __riscv_vsetvl_e16m1(n);
//     vfloat16m1_t vec_x = __riscv_vfmv_v_f_f16m1(scalar, vl);
//     __riscv_vsse16_v_f16m1(x, sizeof(float16_t) * incx, vec_x, vl);
//     x += vl;
//     n -= vl;
//   }
// }

void NN__fill_f32(size_t n, float *x, size_t incx, float scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vfmv_v_f_f32m1(scalar, vl);
    __riscv_vsse32_v_f32m1(x, sizeof(float) * incx, vec_x, vl);
    x += vl;
    n -= vl;
  }
}


#endif
