
#include "nn_maximum.h"
#include "riscv_vector.h"

void NN_maximum_F32_RVV(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(b->size == a->size);
  assert(out->size == a->size);

  float *ptr_out = out->data;
  float *ptr_a = a->data;
  float *ptr_b = b->data;

  size_t n = out->shape[0] * out->shape[1];
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
    vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(ptr_b, vl);
    vfloat32m1_t vec_out = __riscv_vfmax_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(ptr_out, vec_out, vl);
    ptr_a += vl;
    ptr_b += vl;
    ptr_out += vl;
    n -= vl;
  }
}

