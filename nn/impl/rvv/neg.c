#include <riscv_vector.h>
#include "neg.h"

#ifdef RVV


void NN__neg_f32(size_t n, float *y, size_t incy, float *x, size_t incx) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, incx, vl);
    vfloat32m1_t vec_y = __riscv_vfneg_v_f32m1(vec_x, vl);
    __riscv_vsse32_v_f32m1(y, incy, vec_y, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}


#endif
