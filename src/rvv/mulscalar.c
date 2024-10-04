#include <riscv_vector.h>
#include "nn.h"

#ifdef RISCV_V

#ifdef RISCV_ZVFH
  void NN_mulscalar1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x, float16_t scalar) {
    NN_assert(x->shape[0] == y->shape[0], "Cannot add tensors of different shapes");

    size_t n = y->shape[0];
    float16_t *y_data = y->data;
    float16_t *x_data = x->data;
    
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vfloat16m1_t vec_y = __riscv_vfmul_vf_f16m1(vec_x, scalar, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      y_data += vl;
      x_data += vl;
      n -= vl;
    }
  }
#endif

void NN_mulscalar1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, float scalar) {
  NN_assert(x->shape[0] == y->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float *y_data = y->data;
  float *x_data = x->data;

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
    vfloat32m1_t vec_y = __riscv_vfmul_vf_f32m1(vec_x, scalar, vl);
    __riscv_vse32_v_f32m1(y_data, vec_y, vl);
    y_data += vl;
    x_data += vl;
    n -= vl;
  }
}

#ifdef RISCV_ZVFH
  void NN_mulscalar2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float16_t scalar) {
    NN_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot add tensors of different shapes");

    size_t n = y->shape[0] * y->shape[1];
    float16_t *y_data = y->data;
    float16_t *x_data = x->data;

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vfloat16m1_t vec_y = __riscv_vfmul_vf_f16m1(vec_x, scalar, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      y_data += vl;
      x_data += vl;
      n -= vl;
    }
  }
#endif

void NN_mulscalar2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float scalar) {
  NN_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float *y_data = y->data;
  float *x_data = x->data;
  
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
    vfloat32m1_t vec_y = __riscv_vfmul_vf_f32m1(vec_x, scalar, vl);
    __riscv_vse32_v_f32m1(y_data, vec_y, vl);
    y_data += vl;
    x_data += vl;
    n -= vl;
  }
}

#endif
