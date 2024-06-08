
#include "nn_matmul.h"
#include "riscv_vector.h"

void NN_matmul_F32_RVV(Tensor *out, Tensor *a, Tensor *b) {
  // currently only support 2D matrix multiplication
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(b->shape[0] == a->shape[1]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[1]);

  size_t vlmax = __riscv_vsetvlmax_e32m1();

  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);

  for (size_t i = 0; i < a->shape[0]; i += 1) {
    for (size_t j = 0; j < b->shape[1]; j += 1) {
      
      size_t k = a->shape[1];
      float *a_ptr = ((float *)a->data) + i * k;
      float *b_ptr = ((float *)b->data) + j;
      
      vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
      
      while (k > 0) {
        size_t vl = __riscv_vsetvl_e32m1(k);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a_ptr, vl);
        vfloat32m1_t vec_b = __riscv_vlse32_v_f32m1(b_ptr, b->shape[1] * sizeof(float), vl);
        vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_a, vec_b, vl);
        
        a_ptr += vl;
        b_ptr += vl;
        k -= vl;
      }
      vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
      ((float *)out->data)[i * out->shape[1] + j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    }
  }  
}

void NN_matmult_F32_RVV(Tensor *out, Tensor *a, Tensor *b) {
  // currently only support 2D matrix multiplication
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(a->shape[1] == b->shape[1]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[0]);

  size_t vlmax = __riscv_vsetvlmax_e32m1();

  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);

  for (size_t i = 0; i < a->shape[0]; i += 1) {
    for (size_t j = 0; j < b->shape[1]; j += 1) {
      
      size_t k = a->shape[1];
      float *a_ptr = ((float *)a->data) + i * k;
      float *b_ptr = ((float *)b->data) + j * k;
      
      vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
      
      while (k > 0) {
        size_t vl = __riscv_vsetvl_e32m1(k);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a_ptr, vl);
        vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b_ptr, vl);
        vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_a, vec_b, vl);
        
        a_ptr += vl;
        b_ptr += vl;
        k -= vl;
      }
      vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
      ((float *)out->data)[i * out->shape[1] + j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    }
  }  
}
