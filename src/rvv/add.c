#include <riscv_vector.h>
#include "nn.h"

#ifdef RVV

void NN_add1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  NN_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = x1->shape[0];
  float *x1_data = x1->data;
  float *x2_data = x2->data;
  float *y_data = y->data;
  
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_data, vl);
    vfloat32m1_t vec_x2 = __riscv_vle32_v_f32m1(x2_data, vl);
    vfloat32m1_t vec_y = __riscv_vfadd_vv_f32m1(vec_x1, vec_x2, vl);
    __riscv_vse32_v_f32m1(y_data, vec_y, vl);
    x1_data += vl;
    x2_data += vl;
    y_data += vl;
    n -= vl;
  }
}

void NN_add2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) {
  NN_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = x1->shape[0] * x1->shape[1];
  float *x1_data = x1->data;
  float *x2_data = x2->data;
  float *y_data = y->data;
  
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_data, vl);
    vfloat32m1_t vec_x2 = __riscv_vle32_v_f32m1(x2_data, vl);
    vfloat32m1_t vec_y = __riscv_vfadd_vv_f32m1(vec_x1, vec_x2, vl);
    __riscv_vse32_v_f32m1(y_data, vec_y, vl);
    x1_data += vl;
    x2_data += vl;
    y_data += vl;
    n -= vl;
  }
}

#endif
