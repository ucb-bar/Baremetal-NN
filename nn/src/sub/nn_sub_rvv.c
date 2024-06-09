
#include "nn_sub.h"
#include "riscv_vector.h"

void NN_sub_F32_RVV(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  
  float *out_ptr = out->data;
  float *a_ptr = a->data;
  float *b_ptr = b->data;
  
  // TODO: currently only support 2dim
  assert(a->ndim == 2);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == a->shape[1]);

  // TODO: add broadcasting support
  size_t n = out->shape[0] * out->shape[1];
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a_ptr, vl);
    vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b_ptr, vl);
    vfloat32m1_t vec_out = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(out_ptr, vec_out, vl);
    a_ptr += vl;
    b_ptr += vl;
    out_ptr += vl;
    n -= vl;
  }

  return;
}
