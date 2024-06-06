
#include "nn_max.h"
#include "riscv_vector.h"

float NN_max_F32_RVV(Tensor *t) {
  assert(t->dtype == DTYPE_F32);

  float max = -FLT_MAX;
  float *t_data = (float *)t->data;

  vfloat32m1_t vec_max = __riscv_vfmv_s_f_f32m1(max, 1);

  size_t n = t->shape[0] * t->shape[1];
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_data = __riscv_vle32_v_f32m1(t_data, vl);
    vec_max = __riscv_vfredmax_vs_f32m1_f32m1(vec_data, vec_max, vl);
    t_data += vl;
    n -= vl;
  }
  max = __riscv_vfmv_f_s_f32m1_f32(vec_max);

  return max;
}

