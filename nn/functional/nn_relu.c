
#include "nn_relu.h"


void NN_relu(Tensor *y, Tensor *x) {
  assert(y->ndim == x->ndim);
  assert(y->dtype == x->dtype);
  assert(y->size == x->size);

  switch (y->dtype) {
    case DTYPE_F16:
      NN__maximum1_f16(y->size, (float16_t *)y->data, 1, (float16_t *)x->data, 1, 0.0f);
      return;
    case DTYPE_F32:
      NN__maximum1_f32(y->size, (float *)y->data, 1, (float *)x->data, 1, 0.0f);
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
