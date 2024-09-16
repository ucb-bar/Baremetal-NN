#include "nn.h"


__attribute__((weak)) void NN_add1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  NN_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = x1->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x1->data[i] + x2->data[i]; 
  }
}

__attribute__((weak)) void NN_add2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) {
  NN_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = x1->shape[0] * x1->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x1->data[i] + x2->data[i]; 
  }
}

