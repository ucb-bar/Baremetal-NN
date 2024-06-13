
#include "nn_minimum.h"


void NN_minimum(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(b->size == a->size);
  assert(out->size == a->size);

  switch (out->dtype) {
    case DTYPE_F32:
      #ifdef RVV
        float *ptr_out = out->data;
        float *ptr_a = a->data;
        float *ptr_b = b->data;

        size_t n = out->size;
        while (n > 0) {
          size_t vl = __riscv_vsetvl_e32m1(n);
          vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
          vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(ptr_b, vl);
          vfloat32m1_t vec_out = __riscv_vfmin_vv_f32m1(vec_a, vec_b, vl);
          __riscv_vse32_v_f32m1(ptr_out, vec_out, vl);
          ptr_a += vl;
          ptr_b += vl;
          ptr_out += vl;
          n -= vl;
        }
      #else
        for (size_t i = 0; i < out->size; i += 1) {
          float a_val = ((float *)a->data)[i];
          float b_val = ((float *)b->data)[i];
          ((float *)out->data)[i] = a_val < b_val ? a_val : b_val;
        }
      #endif
      return;

    default:
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = max(%s, %s)\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype)
  );
}
