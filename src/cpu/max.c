#include "nn.h"


__attribute__((weak)) void NN_max1d_f16(Tensor0D_F16 *y, const Tensor1D_F16 *x) {
  y->data = -FLT_MAX;
  size_t n = x->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    float val = as_f32(x->data[i]);
    y->data = val > y->data ? val : y->data;
  }
  return y->data;
}

__attribute__((weak)) void NN_max1d_f32(Tensor0D_F32 *y, const Tensor1D_F32 *x) {
  y->data = -FLT_MAX;
  size_t n = x->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    float val = x->data[i];
    y->data = val > y->data ? val : y->data;
  }
  return y->data;
}

__attribute__((weak)) void NN_max2d_f16(Tensor0D_F16 *y, const Tensor2D_F16 *x) {
  y->data = -FLT_MAX;
  size_t n = x->shape[0] * x->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float val = as_f32(x->data[i]);
    y->data = val > y->data ? val : y->data;
  }
  return y->data;
}

__attribute__((weak)) void NN_max2d_f32(Tensor0D_F32 *y, const Tensor2D_F32 *x) {
  y->data = -FLT_MAX;
  size_t n = x->shape[0] * x->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float val = x->data[i];
    y->data = val > y->data ? val : y->data;
  }
  return y->data;
}
