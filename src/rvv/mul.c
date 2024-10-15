#include <riscv_vector.h>
#include "nn.h"

#ifdef RISCV_V

#ifdef RISCV_ZVFH
  void nn_mul_f16_asm(size_t n, float16_t *y_data, const float16_t *x1_data, const float16_t *x2_data);
#endif
void nn_mul_f32_asm(size_t n, float *y_data, const float *x1_data, const float *x2_data);

#ifdef RISCV_ZVFH
  void nn_mul1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x1, const Tensor1D_F16 *x2) {
    nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
    nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

    size_t n = y->shape[0];
    float16_t *x1_data = x1->data;
    float16_t *x2_data = x2->data;
    float16_t *y_data = y->data;

    #ifdef RISCV_V_ASM
      nn_mul_f16_asm(n, y_data, x1_data, x2_data);
    #else
      while (n > 0) {
        size_t vl = __riscv_vsetvl_e16m1(n);
        vfloat16m1_t vec_x1 = __riscv_vle16_v_f16m1(x1_data, vl);
        vfloat16m1_t vec_x2 = __riscv_vle16_v_f16m1(x2_data, vl);
        vfloat16m1_t vec_y = __riscv_vfmul_vv_f16m1(vec_x1, vec_x2, vl);
        __riscv_vse16_v_f16m1(y_data, vec_y, vl);
        x1_data += vl;
        x2_data += vl;
        y_data += vl;
        n -= vl;
      }
    #endif
  }
#endif

void nn_mul1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float *x1_data = x1->data;
  float *x2_data = x2->data;
  float *y_data = y->data;

  #ifdef RISCV_V_ASM
    nn_mul_f32_asm(n, y_data, x1_data, x2_data);
  #else
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_data, vl);
      vfloat32m1_t vec_x2 = __riscv_vle32_v_f32m1(x2_data, vl);
      vfloat32m1_t vec_y = __riscv_vfmul_vv_f32m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #endif
}

#ifdef RISCV_ZVFH
  void nn_mul2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) {
    nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
    nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

    size_t n = y->shape[0] * y->shape[1];
    float16_t *x1_data = x1->data;
    float16_t *x2_data = x2->data;
    float16_t *y_data = y->data;

    #ifdef RISCV_V_ASM
      nn_mul_f16_asm(n, y_data, x1_data, x2_data);
    #else
      while (n > 0) {
        size_t vl = __riscv_vsetvl_e16m1(n);
        vfloat16m1_t vec_x1 = __riscv_vle16_v_f16m1(x1_data, vl);
        vfloat16m1_t vec_x2 = __riscv_vle16_v_f16m1(x2_data, vl);
        vfloat16m1_t vec_y = __riscv_vfmul_vv_f16m1(vec_x1, vec_x2, vl);
        __riscv_vse16_v_f16m1(y_data, vec_y, vl);
        x1_data += vl;
        x2_data += vl;
        y_data += vl;
        n -= vl;
      }
    #endif
  }
#endif

void nn_mul2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float *x1_data = x1->data;
  float *x2_data = x2->data;
  float *y_data = y->data;
  
  #ifdef RISCV_V_ASM
    nn_mul_f32_asm(n, y_data, x1_data, x2_data);
  #else
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_data, vl);
      vfloat32m1_t vec_x2 = __riscv_vle32_v_f32m1(x2_data, vl);
      vfloat32m1_t vec_y = __riscv_vfmul_vv_f32m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #endif
}

#endif
