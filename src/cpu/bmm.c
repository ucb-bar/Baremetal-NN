#include "nn.h"


__attribute__((weak)) void nn_bmm_f16(Tensor3D_F16 *y, const Tensor3D_F16 *x1, const Tensor3D_F16 *x2) { 
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot perform MatMul on tensors of different shapes");

  const size_t batch_size = x1->shape[0];

  for (size_t i = 0; i < batch_size; i += 1) {
    Tensor2D_F16 yi = {.shape = {x2->shape[0], x2->shape[1]}, .data = y->data + i * x2->shape[0] * x2->shape[1]};
    Tensor2D_F16 x1i = {.shape = {x1->shape[1], x1->shape[2]}, .data = x1->data + i * x1->shape[1] * x1->shape[2]};
    Tensor2D_F16 x2i = {.shape = {x2->shape[1], x2->shape[2]}, .data = x2->data + i * x2->shape[1] * x2->shape[2]};

    nn_mm_f16(&yi, &x1i, &x2i);
  }
}

__attribute__((weak)) void nn_bmm_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x1, const Tensor3D_F32 *x2) { 
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot perform MatMul on tensors of different shapes");

  const size_t batch_size = x1->shape[0];

  for (size_t i = 0; i < batch_size; i += 1) {
    Tensor2D_F32 yi = {.shape = {x2->shape[0], x2->shape[1]}, .data = y->data + i * x2->shape[0] * x2->shape[1]};
    Tensor2D_F32 x1i = {.shape = {x1->shape[1], x1->shape[2]}, .data = x1->data + i * x1->shape[1] * x1->shape[2]};
    Tensor2D_F32 x2i = {.shape = {x2->shape[1], x2->shape[2]}, .data = x2->data + i * x2->shape[1] * x2->shape[2]};

    nn_mm_f32(&yi, &x1i, &x2i);
  }
}