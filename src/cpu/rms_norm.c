#include "nn.h"


__attribute__((weak)) void nn_rms_norm1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, const Tensor1D_F32 *weight, float eps) {
  const size_t n = x->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] * (1.0f / sqrtf(x->data[i] * x->data[i] + eps));
  }
}
