
#include "nn_mul.h"

void NN_mul(Tensor *out, const Tensor *a, const Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(b->dtype == a->dtype);
  assert(out->dtype == a->dtype);
  assert(b->size == a->size);
  assert(out->size == a->size);
  
  switch (out->dtype) {
    case DTYPE_F32:
      NN__mul_f32(out->size, (float *)out->data, 1, (float *)a->data, 1, (float *)b->data, 1);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = %s * %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

void NN_mul1(Tensor *out, const Tensor *in, float scalar) {
  assert(out->ndim == in->ndim);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == in->size);
  
  switch (out->dtype) {
    case DTYPE_F32:
      NN__mul1_f32(out->size, (float *)out->data, 1, (float *)in->data, 1, scalar);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = %s * scalar\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(in->dtype)
  );
}

void NN_mul_inplace(Tensor *b, const Tensor *a) {
  NN_mul(b, b, a);
}

void NN_mul1_inplace(Tensor *b, float scalar) {
  NN_mul1(b, b, scalar);
}

