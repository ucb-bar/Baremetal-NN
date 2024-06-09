
#include "nn_abs.h"
#include "riscv_vector.h"

void NN_abs_F32_RVV(Tensor *out, Tensor *input) {
  assert(out->ndim == input->ndim);
  assert(input->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == input->size);

  float *ptr_out = out->data;
  float *ptr_in = input->data;
  
  size_t n = out->shape[0] * out->shape[1];
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_in = __riscv_vle32_v_f32m1(ptr_in, vl);
    vfloat32m1_t vec_out = __riscv_vfabs_v_f32m1(vec_in, vl);
    __riscv_vse32_v_f32m1(ptr_out, vec_out, vl);
    ptr_in += vl;
    ptr_out += vl;
    n -= vl;
  }
}

