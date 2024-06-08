
#include "nn_relu.h"

void NN_relu_F32(Tensor *y, Tensor *x) {
  assert(x->ndim == 2);
  assert(y->ndim == 2);
  assert(x->dtype == DTYPE_F32);
  assert(y->dtype == DTYPE_F32);
  assert(y->shape[0] == x->shape[0]);
  assert(y->shape[1] == x->shape[1]);

  float *y_data = (float *)y->data;
  float *x_data = (float *)x->data;

  for (size_t i = 0; i < y->shape[0] * y->shape[1]; i+=1) {
    y_data[i] = x_data[i] > 0 ? x_data[i] : 0;
  }
}
