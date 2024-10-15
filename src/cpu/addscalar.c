#include "nn.h"


__attribute__((weak)) void nn_addscalar1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x->data[i]) + as_f32(scalar));
  }
}

__attribute__((weak)) void nn_addscalar1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] + scalar; 
  }
}

__attribute__((weak)) void nn_addscalar2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x->data[i]) + as_f32(scalar));
  }
}

__attribute__((weak)) void nn_addscalar2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] + scalar; 
  }
}

