
#include "nn_relu.h"


void NN_ReLU6(Tensor *y, Tensor *x) {
  assert(y->ndim == x->ndim);
  assert(y->dtype == x->dtype);
  assert(y->size == x->size);
  
  switch (y->dtype) {
    case DTYPE_F32:
      NN__maximum1_F32(y->size, (float *)y->data, (float *)x->data, 0.0f);
      NN__minimum1_F32(y->size, (float *)y->data, (float *)y->data, 6.0f);
      return;

    default:
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = ReLU(%s)\n", 
    NN_getDataTypeName(y->dtype), NN_getDataTypeName(x->dtype)
  );
}

void NN_ReLU6Inplace(Tensor *x) {
  NN_ReLU6(x, x);
}
