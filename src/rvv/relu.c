#include <riscv_vector.h>
#include "nn.h"

#ifdef RVV

void NN_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x) {
  NN_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  size_t n = x->shape[0] * x->shape[1];
  float *x_data = x->data;
  float *y_data = y->data;

  float zero = 0.0f;
  
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
    vfloat32m1_t vec_y = __riscv_vfmax_vf_f32m1(vec_x, zero, vl);
    __riscv_vse32_v_f32m1(y_data, vec_y, vl);
    x_data += vl;
    y_data += vl;
    n -= vl;
  } 
}

#endif
