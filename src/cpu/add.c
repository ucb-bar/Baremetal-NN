#include "nn.h"


__attribute__((weak)) void NN_add1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x1, const Tensor1D_F16 *x2) {
  NN_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  float16_t *y_dim0 = y->data;
  float16_t *x1_dim0 = x1->data;
  float16_t *x2_dim0 = x2->data;
  for (size_t i = 0; i < x1->shape[0]; i += 1) {
    *y_dim0 = as_f16(as_f32(*x1_dim0) + as_f32(*x2_dim0));
    y_dim0 += y->stride[0];
    x1_dim0 += x1->stride[0];
    x2_dim0 += x2->stride[0];
  }
}

__attribute__((weak)) void NN_add1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  NN_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  float *y_dim0 = y->data;
  float *x1_dim0 = x1->data;
  float *x2_dim0 = x2->data;
  for (size_t i = 0; i < x1->shape[0]; i += 1) {
    *y_dim0 = *x1_dim0 + *x2_dim0;
    y_dim0 += y->stride[0];
    x1_dim0 += x1->stride[0];
    x2_dim0 += x2->stride[0];
  }
}

__attribute__((weak)) void NN_add2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) {
  NN_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  float16_t *y_dim0 = y->data;
  float16_t *x1_dim0 = x1->data;
  float16_t *x2_dim0 = x2->data;
  for (size_t i = 0; i < x1->shape[0]; i += 1) {
    float16_t *y_dim1 = y_dim0;
    float16_t *x1_dim1 = x1_dim0;
    float16_t *x2_dim1 = x2_dim0;
    for (size_t j = 0; j < x1->shape[1]; j += 1) {
      *y_dim1 = as_f16(as_f32(*x1_dim1) + as_f32(*x2_dim1));
      y_dim1 += y->stride[1];
      x1_dim1 += x1->stride[1];
      x2_dim1 += x2->stride[1];
    }
    y_dim0 += y->stride[0];
    x1_dim0 += x1->stride[0];
    x2_dim0 += x2->stride[0];
  }
}

__attribute__((weak)) void NN_add2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) {
  NN_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  NN_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  float *y_dim0 = y->data;
  float *x1_dim0 = x1->data;
  float *x2_dim0 = x2->data;
  for (size_t i = 0; i < x1->shape[0]; i += 1) {
    float *y_dim1 = y_dim0;
    float *x1_dim1 = x1_dim0;
    float *x2_dim1 = x2_dim0;
    for (size_t j = 0; j < x1->shape[1]; j += 1) {
      *y_dim1 = *x1_dim1 + *x2_dim1;
      y_dim1 += y->stride[1];
      x1_dim1 += x1->stride[1];
      x2_dim1 += x2->stride[1];
    }
    y_dim0 += y->stride[0];
    x1_dim0 += x1->stride[0];
    x2_dim0 += x2->stride[0];
  }
}

