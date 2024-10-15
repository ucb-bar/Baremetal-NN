#include "nn.h"

__attribute__((weak)) void nn_dot_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x1, const Tensor1D_F16 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot dot tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot dot tensors of different shapes");

  size_t n = y->shape[0];
  float sum_f32 = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum_f32 += as_f32(x1->data[i]) * as_f32(x2->data[i]);
  }
  y->data[0] = as_f16(sum_f32);
}

__attribute__((weak)) void nn_dot_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot dot tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot dot tensors of different shapes");

  size_t n = y->shape[0];
  float sum = 0.0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x1->data[i] * x2->data[i];
  }
  y->data[0] = sum;
}
