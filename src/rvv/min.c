#include <riscv_vector.h>
#include "nn.h"

#ifdef RISCV_ZVFH
  void nn_min1d_f16(Tensor0D_F16 *y, const Tensor1D_F16 *x) {
    vfloat16m1_t vec_min = __riscv_vfmv_v_f_f16m1(FLT_MAX, 1);
    size_t n = x->shape[0];
    float16_t *x_data = x->data;

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vec_min = __riscv_vfredmin_vs_f16m1_f16m1(vec_x, vec_min, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f16m1_f16(vec_min);
  }
#endif

void nn_min1d_f32(Tensor0D_F32 *y, const Tensor1D_F32 *x) {
  vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(FLT_MAX, 1);
  size_t n = x->shape[0];
  float *x_data = x->data;

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
    vec_min = __riscv_vfredmin_vs_f32m1_f32m1(vec_x, vec_min, vl);
    x_data += vl;
    n -= vl;
  }
  y->data = __riscv_vfmv_f_s_f32m1_f32(vec_min);
}

#ifdef RISCV_ZVFH
  void nn_min2d_f16(Tensor0D_F16 *y, const Tensor2D_F16 *x) {
    vfloat16m1_t vec_min = __riscv_vfmv_v_f_f16m1(FLT_MAX, 1);
    size_t n = x->shape[0] * x->shape[1];
    float16_t *x_data = x->data;

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vec_min = __riscv_vfredmin_vs_f16m1_f16m1(vec_x, vec_min, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f16m1_f16(vec_min);
  }
#endif

void nn_min2d_f32(Tensor0D_F32 *y, const Tensor2D_F32 *x) {
  vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(FLT_MAX, 1);
  size_t n = x->shape[0] * x->shape[1];
  float *x_data = x->data;

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
    vec_min = __riscv_vfredmin_vs_f32m1_f32m1(vec_x, vec_min, vl);
    x_data += vl;
    n -= vl;
  }
  y->data = __riscv_vfmv_f_s_f32m1_f32(vec_min);
}

