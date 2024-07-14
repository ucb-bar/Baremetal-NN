#include <riscv_vector.h>
#include "neg.h"

#ifdef RVV


void NN__transpose_f32(size_t m, size_t n, float *y, const float *x) {
  for (size_t i = 0; i < m; i += 1) {
    size_t k = n;
    while (k > 0) {
      size_t vl = __riscv_vsetvl_e32m1(k);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
      __riscv_vsse32_v_f32m1(y, sizeof(float) * m, vec_x, vl);
      x += vl;
      y += vl * m;
      k -= vl;
    }
  }
}


#endif
