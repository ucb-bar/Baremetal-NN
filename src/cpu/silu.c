#include "nn.h"


__attribute__((weak)) void nn_silu1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x) {
  // nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform SiLU on tensors of different shapes");

}

__attribute__((weak)) void nn_silu1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x) {
  // nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform SiLU on tensors of different shapes");

  const size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] * (1.0f / (1.0f + expf(-x->data[i])));
  }
}

__attribute__((weak)) void nn_silu2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x) {
  // nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform SiLU on tensors of different shapes");

}

__attribute__((weak)) void nn_silu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x) {
  // nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform SiLU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] * (1.0f / (1.0f + expf(-x->data[i])));
  }
}

