#include "nn.h"


__attribute__((weak)) void NN_mm_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) { 
  NN_assert(x1->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");

  const size_t batch_size = x1->shape[0];
  const size_t in_features = x1->shape[1];
  const size_t out_features = x2->shape[0];

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < out_features; j++) {
      float sum = 0.f;
      for (size_t k = 0; k < in_features; k++) {
        sum += as_f32(x1->data[i * in_features + k]) * as_f32(x2->data[j * in_features + k]);
      }
      y->data[i * out_features + j] = as_f16(sum);
    }
  }
}

__attribute__((weak)) void NN_mm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) { 
  NN_assert(x1->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");

  const size_t batch_size = x1->shape[0];
  const size_t in_features = x1->shape[1];
  const size_t out_features = x2->shape[0];

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < out_features; j++) {
      float sum = 0.f;
      for (size_t k = 0; k < in_features; k++) {
        sum += x1->data[i * in_features + k] * x2->data[j * in_features + k];
      }
      y->data[i * out_features + j] = sum;
    }
  }
}
