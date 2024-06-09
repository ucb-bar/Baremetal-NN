
#include "nn_multiply.h"

void NN_multiply_F32(Tensor *out, Tensor *in, float coeff) {
  assert(out->ndim == in->ndim);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == in->size);
  
  for (size_t i = 0; i < out->size; i += 1) {
    ((float *)out->data)[i] = ((float *)in->data)[i] * coeff;
  }
}

