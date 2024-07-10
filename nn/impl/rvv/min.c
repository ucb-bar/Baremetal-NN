#include <riscv_vector.h>
#include "min.h"

#ifdef RVV


void NN__min_f32(size_t n, float *result, float *x, size_t incx) {
  vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(FLT_MAX, 1);
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, sizeof(float) * incx, vl);
    vec_min = __riscv_vfredmin_vs_f32m1_f32m1(vec_x, vec_min, vl);
    x += vl;
    n -= vl;
  }
  *result = __riscv_vfmv_f_s_f32m1_f32(vec_min);
}


#endif
