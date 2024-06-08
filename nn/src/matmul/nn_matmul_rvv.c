
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

  uint8_t *out_ptr = out->data;
  uint8_t *a_ptr = a->data;
  uint8_t *b_ptr = b->data;

  size_t vlmax = __riscv_vsetvlmax_e32m1();

  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
  for (size_t i = 0; i < a->shape[0]; i += 1) {
    for (size_t j = 0; j < b->shape[1]; j += 1) {
      uint8_t *a_ptr_v = a_ptr;
      uint8_t *b_ptr_v = b_ptr;
      
      vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
      
      size_t n = a->shape[1];
      while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t vec_a = __riscv_vlse32_v_f32m1((float *)a_ptr_v, a->strides[1], vl);
        vfloat32m1_t vec_b = __riscv_vlse32_v_f32m1((float *)b_ptr_v, b->strides[0], vl);
        vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_a, vec_b, vl);
        
        a_ptr_v += vl * a->strides[1];
        b_ptr_v += vl * b->strides[0];
        n -= vl;
      }
      vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
      *((float *)out_ptr) = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
      
      out_ptr += out->strides[1];
      b_ptr += b->strides[1];
    }
    out_ptr -= out->strides[1] * b->shape[1];
    out_ptr += out->strides[0]; // Move to the next row in the output matrix
    a_ptr += a->strides[0];
    b_ptr -= b->strides[1] * b->shape[1];
  }  
}
