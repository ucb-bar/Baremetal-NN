#include <riscv_vector.h>
#include "acc1.h"

#ifdef RVV

void NN__acc1_f32(size_t n, float *result, size_t incx, float scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_r = __riscv_vlse32_v_f32m1(result, incx, vl);
    vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(scalar, vl);
    vec_r = __riscv_vfadd_vv_f32m1(vec_r, vec_s, vl);
    __riscv_vse32_v_f32m1(result, vec_r, vl);
    result += vl;
    n -= vl;
  }
}

#endif