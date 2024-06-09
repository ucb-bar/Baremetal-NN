
#include "nn_minimum.h"


void NN_minimum_F32(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(b->size == a->size);
  assert(out->size == a->size);
  
  for (size_t i = 0; i < out->size; i += 1) {
    float a_val = ((float *)a->data)[i];
    float b_val = ((float *)b->data)[i];
    ((float *)out->data)[i] = a_val < b_val ? a_val : b_val;
  }
}
