
#include "nn_relu.h"


void NN_relu(Tensor *y, Tensor *x) {
  assert(y->ndim == x->ndim);
  assert(y->dtype == x->dtype);
  assert(y->size == x->size);

  switch (y->dtype) {
    case DTYPE_F32:
      NN__maximum1_f32(y->size, (float *)y->data, (float *)x->data, 0.0f);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = ReLU(%s)\n", 
    NN_get_datatype_name(y->dtype), NN_get_datatype_name(x->dtype)
  );
}

void NN_relu_inplace(Tensor *x) {
  NN_relu(x, x);
}