
#include "nn_relu6.h"

void NN_ReLU6_F32(Tensor *y, Tensor *x) {
  assert(y->ndim == x->ndim);
  assert(x->dtype == DTYPE_F32);
  assert(y->dtype == DTYPE_F32);
  assert(y->size == x->size);

  for (size_t i = 0; i < x->size; i += 1) {
    float val = ((float *)x->data)[i];
    ((float *)y->data)[i] = val > 0 ? (val < 6 ? val : 6) : 0;
  }
}

void NN_ReLU6Inplace_F32(Tensor *x) {
  for (size_t i = 0; i < x->size; i += 1) {
    float val = ((float *)x->data)[i];
    ((float *)x->data)[i] = val > 0 ? (val < 6 ? val : 6) : 0;
  }
}
