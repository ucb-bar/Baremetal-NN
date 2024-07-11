#include <riscv_vector.h>
#include "add.h"

#ifdef RVV


void NN__add_i8(size_t n, int8_t *z, size_t incz, int8_t *x, size_t incx, int8_t *y, size_t incy) {
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

void NN__add_f16(size_t n, float16_t *z, size_t incz, float16_t *x, size_t incx, float16_t *y, size_t incy) {
  while (n > 0) {
    size_t vl;
    
    // size_t vl = __riscv_vsetvl_e16m1(n);
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(n));

    // vfloat16m1_t vec_x = __riscv_vlse16_v_f16m1(x, sizeof(float16_t) * incx, vl);
    asm volatile("vlse16.v v24, (%0), %1" : : "r"(x), "r"(sizeof(float16_t) * incx));

    // vfloat16m1_t vec_y = __riscv_vlse16_v_f16m1(y, sizeof(float16_t) * incy, vl);
    asm volatile("vlse16.v v25, (%0), %1" : : "r"(y), "r"(sizeof(float16_t) * incy));
    
    // // vfloat16m1_t vec_z = __riscv_vfadd_vv_f16m1(vec_x, vec_y, vl);
    asm volatile("vfadd.vv v24, v24, v25");

    // __riscv_vsse16_v_f16m1(z, sizeof(float16_t) * incz, vec_z, vl);
    asm volatile("vsse16.v v24, (%0), %1" : : "r"(z), "r"(sizeof(float16_t) * incz));

    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}

void NN__add_f32(size_t n, float *z, size_t incz, float *x, size_t incx, float *y, size_t incy) {
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
