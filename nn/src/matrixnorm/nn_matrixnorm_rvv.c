
#include "nn_matrixnorm.h"
#include "riscv_vector.h"


float NN_matrixNorm_F32_RVV(Tensor *tensor) {
  assert(tensor->ndim == 2);
  assert(tensor->dtype == DTYPE_F32);
  
  float *ptr = tensor->data;

  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
  vfloat32m1_t vec_accumulate = __riscv_vfmv_v_f_f32m1(0, vlmax);

  size_t n = tensor->shape[0] * tensor->shape[1];
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr, vl);
    vec_accumulate = __riscv_vfmacc_vv_f32m1(vec_accumulate, vec_a, vec_a, vl);
    ptr += vl;
    n -= vl;
  }
  vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_accumulate, vec_zero, vlmax);
  float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);

  return sqrt(sum);
}
