
#include "nn_mul.h"
#include "riscv_vector.h"

void NN_mul_F32_RVV(Tensor *out, Tensor *in, float scalar) {
  assert(out->ndim == in->ndim);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  
  uint8_t *in_ptr = in->data;
  uint8_t *out_ptr = out->data;
  
  // TODO: currently only support 2dim
  assert(in->ndim == 2);
  assert(out->shape[0] == in->shape[0]);
  assert(out->shape[1] == in->shape[1]);
  
  size_t n = in->shape[0] * in->shape[1];
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_in = __riscv_vlse32_v_f32m1((float *)in_ptr, in->strides[1], vl);
    vfloat32m1_t vec_out = __riscv_vfmul_vf_f32m1(vec_in, scalar, vl);
    __riscv_vsse32_v_f32m1((float *)out_ptr, out->strides[1], vec_out, vl);
    in_ptr += vl * in->strides[1];
    out_ptr += vl * out->strides[1];
    n -= vl;
  }
}
