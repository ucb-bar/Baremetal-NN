#include "nn.h"


__attribute__((weak)) void nn_softmax1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, size_t dim) {
  const size_t n = x->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = expf(x->data[i]);
  }
}
