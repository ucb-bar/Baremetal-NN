#include <riscv_vector.h>
#include "nn.h"

#ifdef RISCV_V

#ifdef RISCV_ZVFH
  void NN_relu_f16_asm(size_t n, float16_t *y_data, const float16_t *x_data);
#endif
void NN_relu_f32_asm(size_t n, float *y_data, const float *x_data);

#ifdef RISCV_ZVFH
  void NN_relu2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x) {
    NN_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

    size_t n = y->shape[0] * y->shape[1];
    float16_t *x_data = x->data;
    float16_t *y_data = y->data;

    #ifdef RISCV_V_ASM
      NN_relu_f16_asm(n, y_data, x_data);
    #else
      float16_t zero = 0.0f;

      while (n > 0) {
        size_t vl = __riscv_vsetvl_e16m1(n);
        vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
        vfloat16m1_t vec_y = __riscv_vfmax_vf_f16m1(vec_x, zero, vl);
        __riscv_vse16_v_f16m1(y_data, vec_y, vl);
        x_data += vl;
        y_data += vl;
        n -= vl;
      }
    #endif
  }
#endif

void NN_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x) {
  NN_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float *x_data = x->data;
  float *y_data = y->data;

  #ifdef RISCV_V_ASM
    NN_relu_f32_asm(n, y_data, x_data);
  #else
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
  #endif
}

#endif
