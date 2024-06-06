
#include "nn_min.h"
#include "riscv_vector.h"

float NN_min_F32_RVV(Tensor *t) {
  assert(t->dtype == DTYPE_F32);

  float min = FLT_MAX;
  float *t_data = (float *)t->data;

  vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(min, 1);
  size_t i = 0;
  size_t vl = 0;
  for (size_t k = t->shape[0] * t->shape[1]; k > 0; k -= vl, i += vl) {
    vl = __riscv_vsetvl_e32m1(k);
    vfloat32m1_t vec_t = __riscv_vle32_v_f32m1(t_data + i, vl);
    vec_min = __riscv_vfredmin_vs_f32m1_f32m1(vec_t, vec_min, vl);
  }
  min = __riscv_vfmv_f_s_f32m1_f32(vec_min);
  return min;
}

