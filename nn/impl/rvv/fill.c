#include <riscv_vector.h>
#include "fill.h"

#ifdef RVV


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
