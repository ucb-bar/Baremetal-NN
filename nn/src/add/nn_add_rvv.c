
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

  // TODO: add broadcasting support
  size_t n = out->shape[0] * out->shape[1];
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a_data, vl);
    vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b_data, vl);
    vfloat32m1_t vec_c = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(out_data, vec_c, vl);
    a_data += vl;
    b_data += vl;
    out_data += vl;
    n -= vl;
  }
}

