
#include "nn_min.h"
#include "riscv_vector.h"

float NN_min_F32_RVV(Tensor *tensor) {
  assert(tensor->dtype == DTYPE_F32);

  float min = FLT_MAX;
  float *ptr = tensor->data;

  vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(min, 1);

  size_t n = tensor->shape[0] * tensor->shape[1];
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_data = __riscv_vle32_v_f32m1(ptr, vl);
    vec_min = __riscv_vfredmin_vs_f32m1_f32m1(vec_data, vec_min, vl);
    ptr += vl;
    n -= vl;
  }
  min = __riscv_vfmv_f_s_f32m1_f32(vec_min);

  return min;
}

