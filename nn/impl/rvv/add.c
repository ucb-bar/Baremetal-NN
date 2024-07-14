#include <riscv_vector.h>
#include "add.h"

#ifdef RVV


void NN__add_i8(size_t n, int8_t *z, size_t incz, const int8_t *x, size_t incx, const int8_t *y, size_t incy) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vint8m1_t vec_x = __riscv_vlse8_v_i8m1(x, sizeof(int8_t) * incx, vl);
    vint8m1_t vec_y = __riscv_vlse8_v_i8m1(y, sizeof(int8_t) * incy, vl);
    vint8m1_t vec_z = __riscv_vadd_vv_i8m1(vec_x, vec_y, vl);
    __riscv_vsse8_v_i8m1(z, sizeof(int8_t) * incz, vec_z, vl);
    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}

// void NN__add_f16(size_t n, float16_t *z, size_t incz, const float16_t *x, size_t incx, const float16_t *y, size_t incy) {
//   while (n > 0) {
//     size_t vl = __riscv_vsetvl_e16m1(n);
//     vfloat16m1_t vec_x = __riscv_vlse16_v_f16m1(x, sizeof(float16_t) * incx, vl);
//     vfloat16m1_t vec_y = __riscv_vlse16_v_f16m1(y, sizeof(float16_t) * incy, vl);
//     vfloat16m1_t vec_z = __riscv_vfadd_vv_f16m1(vec_x, vec_y, vl);
//     __riscv_vsse16_v_f16m1(z, sizeof(float16_t) * incz, vec_z, vl);
//     x += vl;
//     y += vl;
//     z += vl;
//     n -= vl;
//   }
// }

void NN__add_f32(size_t n, float *z, size_t incz, const float *x, size_t incx, const float *y, size_t incy) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, sizeof(float) * incx, vl);
    vfloat32m1_t vec_y = __riscv_vlse32_v_f32m1(y, sizeof(float) * incy, vl);
    vfloat32m1_t vec_z = __riscv_vfadd_vv_f32m1(vec_x, vec_y, vl);
    __riscv_vsse32_v_f32m1(z, sizeof(float) * incz, vec_z, vl);
    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}

#endif
