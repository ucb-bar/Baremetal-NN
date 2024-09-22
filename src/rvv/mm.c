#include <riscv_vector.h>
#include "nn.h"

#ifdef RISCV_V

#ifdef RISCV_ZVFH
  void NN_mm_f16_asm(size_t in_features, size_t out_features, float16_t *y_data, const float16_t *x1_data, const float16_t *x2_data);
#endif
void NN_mm_f32_asm(size_t in_features, size_t out_features, float *y_data, const float *x1_data, const float *x2_data);

#ifdef RISCV_ZVFH
  void NN_mm_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) { 
    NN_assert(x1->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");
    NN_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");

    const size_t batch_size = x1->shape[0];
    const size_t in_features = x1->shape[1];
    const size_t out_features = x2->shape[0];

    float16_t *x1_batch_data = x1->data;
    float16_t *x2_batch_data = x2->data;
    float16_t *y_batch_data = y->data;

    for (size_t i = 0; i < batch_size; i += 1) {
      float16_t *x1_data = x1_batch_data;
      float16_t *x2_data = x2_batch_data;
      float16_t *y_data = y_batch_data;

      #ifdef RISCV_V_ASM
        NN_mm_f16_asm(in_features, out_features, y_data, x1_data, x2_data);
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
            
            x1_data += vl;
            x2_data += vl;
            n -= vl;
          }
          #ifdef DEBUG_USE_REDOSUM
            vec_sum = __riscv_vfredusum_vs_f16m1_f16m1(vec_sum, vec_zero, vlmax);
          #else
            vec_sum = __riscv_vfredosum_vs_f16m1_f16m1(vec_sum, vec_zero, vlmax);
          #endif
          y_data[j] = __riscv_vfmv_f_s_f16m1_f16(vec_sum) + bias_data[j];
          
          x1_data -= in_features;
        }
      #endif

      x1_batch_data += in_features;
      y_batch_data += out_features;
    }
  }
#endif

void NN_mm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) { 
  NN_assert(x1->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");

  const size_t batch_size = x1->shape[0];
  const size_t in_features = x1->shape[1];
  const size_t out_features = x2->shape[0];

  float *x1_batch_data = x1->data;
  float *x2_batch_data = x2->data;
  float *y_batch_data = y->data;

  for (size_t i = 0; i < batch_size; i += 1) {
    float *x1_data = x1_batch_data;
    float *x2_data = x2_batch_data;
    float *y_data = y_batch_data;

    #ifdef RISCV_V_ASM
      NN_mm_f32_asm(in_features, out_features, y_data, x1_data, x2_data);
    #else
      size_t vlmax = __riscv_vsetvlmax_e32m1();

      for (size_t j = 0; j < out_features; j += 1) {
        vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
        vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
        
        size_t n = in_features;
        
        while (n > 0) {
          size_t vl = __riscv_vsetvl_e32m1(n);
          vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x1_data, vl);
          vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(x2_data, vl);
          vec_sum = __riscv_vfmacc_vv_f32m1(vec_sum, vec_x, vec_y, vl);
          
          x1_data += vl;
          x2_data += vl;
          n -= vl;
        }
        #ifdef DEBUG_USE_REDOSUM
          vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #else
          vec_sum = __riscv_vfredosum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #endif
        y_data[j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        
        x1_data -= in_features;
      }
    #endif

    x1_batch_data += in_features;
    y_batch_data += out_features;
  }
}

#endif
