
#include "nn_elu.h"

void NN_elu_F32(Tensor *y, Tensor *x, float alpha) {
  assert(y->shape[0] == x->shape[0]);
  assert(y->shape[1] == x->shape[1]);
  assert(y->dtype == DTYPE_F32);
  assert(x->dtype == DTYPE_F32);

  for (size_t i = 0; i < y->shape[0] * y->shape[1]; i += 1) {
    if (((float *)x->data)[i] > 0) {
      ((float *)y->data)[i] = ((float *)x->data)[i];
    } else {
      ((float *)y->data)[i] = 1.0f * (expf(((float *)x->data)[i]) - 1.f);
    }
  }
}
