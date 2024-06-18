
#include "nn_neg.h"


void NN_neg(Tensor *out, Tensor *in) {
  assert(out->ndim == in->ndim);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == in->size);
  
  switch (out->dtype) {
    case DTYPE_F32:
      NN__neg_F32(out->size, (float *)out->data, (float *)in->data);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = -%s\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(in->dtype)
  );
}

void NN_negInplace(Tensor *tensor) {
  NN_neg(tensor, tensor);
}
