
#include "nn_mul.h"

void NN_mul(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(b->size == a->size);
  assert(out->size == a->size);
  
  switch (out->dtype) {
    case DTYPE_F32:
      NN__mul_F32(out->size, (float *)out->data, (float *)a->data, (float *)b->data);
      return;

    default:
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = %s * %s\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype)
  );
}

void NN_mul1(Tensor *out, Tensor *in, float scalar) {
  assert(out->ndim == in->ndim);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == in->size);
  
  switch (out->dtype) {
    case DTYPE_F32:
      NN__mul1_F32(out->size, (float *)out->data, (float *)in->data, scalar);
      return;

    default:
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = %s * scalar\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(in->dtype)
  );
}

