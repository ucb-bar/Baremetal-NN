
#include "nn_div.h"

void NN_div(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(b->size == a->size);
  assert(out->size == a->size);
  
  switch (out->dtype) {
    case DTYPE_F32:
      NN__div_F32(out->size, (float *)out->data, (float *)a->data, (float *)b->data);
      return;

    default:
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = %s / %s\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype)
  );
}
