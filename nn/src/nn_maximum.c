
#include "nn_maximum.h"


void NN_maximum(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(b->size == a->size);
  assert(out->size == a->size);

  switch (out->dtype) {
    case DTYPE_F32:
      NN__maximum_F32(out->size, (float *)out->data, (float *)a->data, (float *)b->data);
      return;
    default:
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = max(%s, %s)\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype)
  );
}
