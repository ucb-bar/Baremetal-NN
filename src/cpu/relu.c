#include "nn.h"


__attribute__((weak)) void NN_relu2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x) {
  NN_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = as_f32(x->data[i]);
    y->data[i] = x_val > 0 ? as_f16(x_val) : 0;
  }
}

__attribute__((weak)) void NN_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x) {
  NN_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x->data[i];
    y->data[i] = x_val > 0 ? x_val : 0;
  }
}

