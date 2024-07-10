#include <riscv_vector.h>
#include "minimum.h"

#ifdef RVV


void NN__minimum_f32(size_t n, float *z, size_t incz, float *x, size_t incx, float *y, size_t incy) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, incx, vl);
    vfloat32m1_t vec_y = __riscv_vlse32_v_f32m1(y, incy, vl);
    vfloat32m1_t vec_z = __riscv_vfmin_vv_f32m1(vec_x, vec_y, vl);
    __riscv_vsse32_v_f32m1(z, incz, vec_z, vl);
    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}


#endif
