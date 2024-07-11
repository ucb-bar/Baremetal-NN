#include <riscv_vector.h>
#include "maximum1.h"

#ifdef RVV


void NN__maximum1_f16(size_t n, float16_t *y, size_t incy, float16_t *x, size_t incx, float16_t scalar) {
  while (n > 0) {
    size_t vl;
    // size_t vl = __riscv_vsetvl_e16m1(n);
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(n));

    // vfloat16m1_t vec_x = __riscv_vlse16_v_f16m1(x, sizeof(float16_t) * incx, vl);
    asm volatile("vlse16.v v26, (%0), %1" : : "r"(x), "r"(sizeof(float16_t) * incx));

    // vfloat16m1_t vec_s = __riscv_vfmv_v_f_f16m1(scalar, vl);
    asm volatile("vmv.v.x v25, %0" : : "r"(scalar));

    // vfloat16m1_t vec_y = __riscv_vfmax_vv_f16m1(vec_x, vec_s, vl);
    asm volatile("vfmax.vv v25, v26, v25");

    // __riscv_vsse16_v_f16m1(y, sizeof(float16_t) * incy, vec_y, vl);
    asm volatile("vsse16.v v25, (%0), %1" : : "r"(y), "r"(sizeof(float16_t) * incy));

    x += vl;
    y += vl;
    n -= vl;
  }
}

void NN__maximum1_f32(size_t n, float *y, size_t incy, float *x, size_t incx, float scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, sizeof(float) * incx, vl);
    vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(scalar, vl);
    vfloat32m1_t vec_y = __riscv_vfmax_vv_f32m1(vec_x, vec_s, vl);
    __riscv_vsse32_v_f32m1(y, sizeof(float) * incy, vec_y, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}


#endif
