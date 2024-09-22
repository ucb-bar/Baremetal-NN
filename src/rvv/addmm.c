#include <riscv_vector.h>
#include "nn.h"

#ifdef RISCV_V

#ifdef RISCV_ZVFH
  void NN_addmm_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, const Tensor2D_F16 *weight, const Tensor1D_F16 *bias) { 
    NN_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
    NN_assert(bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
    NN_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = weight->shape[0];

    float16_t *x_batch_data = x->data;
    float16_t *y_batch_data = y->data;

    for (size_t i = 0; i < batch_size; i += 1) {
      float16_t *weight_data = weight->data;
      float16_t *bias_data = bias->data;
      float16_t *x_data = x_batch_data;
      float16_t *y_data = y_batch_data;

      #ifdef RISCV_V_ASM
        NN_addmm_f16_asm(in_features, out_features, y_data, x_data, weight_data, bias_data);
      #else
        size_t vlmax = __riscv_vsetvlmax_e16m1();

        for (size_t j = 0; j < out_features; j += 1) {
          vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
          vfloat16m1_t vec_sum = __riscv_vfmv_v_f_f16m1(0, vlmax);
          
          size_t n = in_features;
          
          while (n > 0) {
            size_t vl = __riscv_vsetvl_e16m1(n);
            vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
            vfloat16m1_t vec_y = __riscv_vle16_v_f16m1(weight_data, vl);
            vec_sum = __riscv_vfmacc_vv_f16m1(vec_sum, vec_x, vec_y, vl);
            
            x_data += vl;
            weight_data += vl;
            n -= vl;
          }
          vec_sum = __riscv_vfredusum_vs_f16m1_f16m1(vec_sum, vec_zero, vlmax);
          y_data[j] = __riscv_vfmv_f_s_f16m1_f16(vec_sum) + bias_data[j];
          
          x_data -= in_features;
        }
      #endif

      x_batch_data += in_features;
      y_batch_data += out_features;
    }
  }
#endif

void NN_addmm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const Tensor2D_F32 *weight, const Tensor1D_F32 *bias) { 
  NN_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  NN_assert(bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  NN_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];

  float *x_batch_data = x->data;
  float *y_batch_data = y->data;

  for (size_t i = 0; i < batch_size; i += 1) {
    float *weight_data = weight->data;
    float *bias_data = bias->data;
    float *x_data = x_batch_data;
    float *y_data = y_batch_data;

    #ifdef RISCV_V_ASM
      NN_addmm_f32_asm(in_features, out_features, y_data, x_data, weight_data, bias_data);
    #else
      size_t vlmax = __riscv_vsetvlmax_e32m1();

      for (size_t j = 0; j < out_features; j += 1) {
        vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
        vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
        
        size_t n = in_features;
        
        while (n > 0) {
          size_t vl = __riscv_vsetvl_e32m1(n);
          vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
          vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(weight_data, vl);
          vec_sum = __riscv_vfmacc_vv_f32m1(vec_sum, vec_x, vec_y, vl);
          
          x_data += vl;
          weight_data += vl;
          n -= vl;
        }
        vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        y_data[j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum) + bias_data[j];
        
        x_data -= in_features;
      }
    #endif

    x_batch_data += in_features;
    y_batch_data += out_features;
  }
}

#endif
