
#include "nn_relu6.h"


void NN_relu6(Tensor *y, const Tensor *x) {
  assert(y->ndim == x->ndim);
  assert(y->dtype == x->dtype);
  assert(y->size == x->size);
  
  switch (y->dtype) {
    case DTYPE_F32:
      NN_maximum1_f32(y->size, (float *)y->data, 1, (float *)x->data, 1, 0.0f);
      NN_minimum1_f32(y->size, (float *)y->data, 1, (float *)y->data, 1, 6.0f);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = ReLU(%s)\n", 
    NN_get_datatype_name(y->dtype), NN_get_datatype_name(x->dtype)
  );
}

void NN_relu6_inplace(Tensor *x) {
  NN_relu6(x, x);
}
