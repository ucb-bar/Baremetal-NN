
#include "nn_add.h"
#include "riscv_vector.h"

void NN_add_F32_RVV(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(a->shape[0] == b->shape[0]);
  
  out->dtype = DTYPE_F32;
  out->shape[0] = a->shape[0];
  out->shape[1] = a->shape[1];
  
  uint8_t *out_ptr = out->data;
  uint8_t *a_ptr = a->data;
  uint8_t *b_ptr = b->data;

  // TODO: add broadcasting support
  size_t n = out->shape[0] * out->shape[1];
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_a = __riscv_vlse32_v_f32m1(a_ptr, a->strides[1], vl);
    vfloat32m1_t vec_b = __riscv_vlse32_v_f32m1(b_ptr, b->strides[1], vl);
    vfloat32m1_t vec_out = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vsse32_v_f32m1(out_ptr, out->strides[1], vec_out, vl);
    a_ptr += vl * a->strides[1];
    b_ptr += vl * b->strides[1];
    out_ptr += vl * out->strides[1];
    n -= vl;
  }

  return;
}

