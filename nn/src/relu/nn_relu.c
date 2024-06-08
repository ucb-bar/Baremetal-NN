
#include "nn_relu.h"

void NN_relu_F32(Tensor *y, Tensor *x) {
  assert(y->ndim == x->ndim);
  assert(x->dtype == DTYPE_F32);
  assert(y->dtype == DTYPE_F32);
  assert(y->size == x->size);

  for (size_t i = 0; i < x->size; i += 1) {
    float val = ((float *)x->data)[i];
    ((float *)y->data)[i] = val > 0 ? val : 0;
  }
}
