#include "nn.h"


__attribute__((weak)) void NN_addmm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const Tensor2D_F32 *weight, const Tensor1D_F32 *bias) { 
  NN_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  NN_assert(bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  NN_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < out_features; j++) {
      float sum = 0.f;
      for (size_t k = 0; k < in_features; k++) {
        sum += x->data[i * in_features + k] * weight->data[j * in_features + k];
      }
      y->data[i * out_features + j] = sum + bias->data[j];
    }
  }
}
