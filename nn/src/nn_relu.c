
#include "nn_relu.h"

void NN_relu_F32(Tensor *y, Tensor *x) {
  assert(y->shape[0] == x->shape[0]);
  assert(y->shape[1] == x->shape[1]);
  assert(y->dtype == DTYPE_F32);
  assert(x->dtype == DTYPE_F32);

  for (size_t i = 0; i < y->shape[0] * y->shape[1]; i++) {
    y->data[i] = x->data[i] > 0 ? x->data[i] : 0;
  }
}
