#include <riscv_vector.h>
#include "maximum.h"

#ifdef RVV


void NN__maximum_f32(size_t n, float *z, size_t incz, const float *x, size_t incx, const float *y, size_t incy) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, sizeof(float) * incx, vl);
    vfloat32m1_t vec_y = __riscv_vlse32_v_f32m1(y, sizeof(float) * incy, vl);
    vfloat32m1_t vec_z = __riscv_vfmax_vv_f32m1(vec_x, vec_y, vl);
    __riscv_vsse32_v_f32m1(z, sizeof(float) * incz, vec_z, vl);
    x += vl;
    y += vl;
    z += vl;
    n -= vl;
  }
}


#endif
