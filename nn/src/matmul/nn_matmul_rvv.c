
#include "nn_matmul.h"
#include "riscv_vector.h"

void NN_matmul_F32_RVV(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(a->shape[1] == b->shape[0]);
  
  out->dtype = DTYPE_F32;
  out->shape[0] = a->shape[0];
  out->shape[1] = b->shape[1];

  size_t vlmax = __riscv_vsetvlmax_e32m1();

  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
  for (int i = 0; i < a->shape[0]; i += 1) {
    for (int j = 0; j < b->shape[1]; j += 1) {
      float *ptr_a = (float *)a->data + i * a->shape[1];
      float *ptr_b = (float *)b->data + j;
      vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
      
      size_t vl = 0;
      for (int k = a->shape[1]; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
        vl = __riscv_vsetvl_e32m1(k);
        vfloat32m1_t vec_a = __riscv_vlse32_v_f32m1(ptr_a, sizeof(float), vl);
        vfloat32m1_t vec_b = __riscv_vlse32_v_f32m1(ptr_b, b->shape[1]*sizeof(float), vl);
        vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_a, vec_b, vl);
      }
      vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
      float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
      ((float *)out->data)[i * out->shape[1] + j] = sum;
    }
  }  
}

