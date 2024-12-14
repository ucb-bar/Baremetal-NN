#include "nn.h"


__attribute__((weak)) void nn_softmax1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float sum = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    sum += expf(as_f32(x->data[i]));
  }

  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(expf(as_f32(x->data[i])) / sum);
  }
}

__attribute__((weak)) void nn_softmax1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float sum = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    sum += expf(x->data[i]);
  }

  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = expf(x->data[i]) / sum;
  }
}

__attribute__((weak)) void nn_softmax2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, size_t dim) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  if (dim == 0) {
    for (size_t i = 0; i < y->shape[0]; i += 1) {
      size_t n = y->shape[1];
      float sum = 0.0f;
      for (size_t j = 0; j < n; j += 1) {
        sum += expf(x->data[i * n + j]);
      }

      for (size_t j = 0; j < n; j += 1) {
        y->data[i * n + j] = expf(x->data[i * n + j]) / sum;
      }
    }
  }
  else if (dim == 1) {
    // HACK: fix batch size
    size_t n = y->shape[1];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i += 1) {
      sum += expf(x->data[i]);
    }

    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = expf(x->data[i]) / sum;
    }
  }
  else {
    nn_assert(0, "Invalid dimension for softmax");
  }
}
