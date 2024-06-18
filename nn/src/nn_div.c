
#include "nn_div.h"

void NN_div(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(b->dtype == a->dtype);
  assert(out->dtype == a->dtype);
  assert(b->size == a->size);
  assert(out->size == a->size);
  
  switch (out->dtype) {
    case DTYPE_F32:
      NN__div_F32(out->size, (float *)out->data, (float *)a->data, (float *)b->data);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = %s / %s\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype)
  );
}
