
#include "nn_add.h"
#include "riscv_vector.h"

void NN_add_F32_RVV(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(a->shape[0] == b->shape[0]);
  
  out->dtype = DTYPE_F32;
  out->shape[0] = a->shape[0];
  out->shape[1] = a->shape[1];
  
  float *out_data = (float *)out->data;
  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;

  int k = out->shape[0] * out->shape[1];
  int l = 0;
  for (size_t vl; k > 0; k -= vl, l += vl) {
    vl = __riscv_vsetvl_e32m1(k);
    vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a_data + l, vl);
    vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b_data + l, vl);
    vfloat32m1_t vec_c = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(out_data + l, vec_c, vl);
  }
}

