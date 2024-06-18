
#include "nn_relu.h"


void NN_ReLU(Tensor *y, Tensor *x) {
  assert(y->ndim == x->ndim);
  assert(y->dtype == x->dtype);
  assert(y->size == x->size);

  switch (y->dtype) {
    case DTYPE_F32:
      NN__maximum1_F32(y->size, (float *)y->data, (float *)x->data, 0.0f);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = ReLU(%s)\n", 
    NN_getDataTypeName(y->dtype), NN_getDataTypeName(x->dtype)
  );
}

void NN_ReLUInplace(Tensor *x) {
  NN_ReLU(x, x);
}
