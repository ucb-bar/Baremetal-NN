#include <riscv_vector.h>
#include "nn.h"

#ifdef RVV

void NN_linear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const Tensor2D_F32 *weight, const Tensor1D_F32 *bias) { 
  NN_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  NN_assert(bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  NN_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];

  float *x_data = x->data;
  float *weight_data = weight->data;
  float *bias_data = bias->data;
  float *y_data = y->data;

  size_t vlmax = __riscv_vsetvlmax_e32m1();

  for (size_t i = 0; i < batch_size; i += 1) {
    for (size_t j = 0; j < out_features; j += 1) {
      vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
      vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
      
      size_t n = in_features;
      float *x_ptr = x_data;
      float *weight_data = weight->data + j * in_features;
      
      while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x_ptr, sizeof(float), vl);
        vfloat32m1_t vec_y = __riscv_vlse32_v_f32m1(weight_data, sizeof(float), vl);
        vec_sum = __riscv_vfmacc_vv_f32m1(vec_sum, vec_x, vec_y, vl);
        
        x_ptr += vl;
        weight_data += vl;
        n -= vl;
      }
      vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
      y_data[j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum) + bias_data[j];
    }
    x_data += in_features;
    y_data += out_features;
  }
}

#endif
