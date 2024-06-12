
#include "nn_mul.h"

void NN_mul_F32(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(b->size == a->size);
  assert(out->size == a->size);
  
  for (size_t i = 0; i < out->size; i += 1) {
    ((float *)out->data)[i] = ((float *)a->data)[i] * ((float *)b->data)[i];
  }
}

void NN_mulF_F32(Tensor *out, Tensor *in, float scalar) {
  assert(out->ndim == in->ndim);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == in->size);
  
  for (size_t i = 0; i < out->size; i += 1) {
    ((float *)out->data)[i] = ((float *)in->data)[i] * scalar;
  }
}

