
#include "nn_neg.h"


void NN_neg(Tensor *out, Tensor *in) {
  assert(out->ndim == in->ndim);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == in->size);
  
  switch (out->dtype) {
    case DTYPE_F32:
      NN__neg_f32(out->size, (float *)out->data, (float *)in->data);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = -%s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(in->dtype)
  );
}

void NN_neg_inplace(Tensor *tensor) {
  NN_neg(tensor, tensor);
}
