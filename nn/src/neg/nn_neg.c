
#include "nn_neg.h"


void NN_neg_F32(Tensor *out, Tensor *input) {
  assert(out->ndim == input->ndim);
  assert(input->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == input->size);
  
  for (size_t i = 0; i < out->size; i += 1) {
    ((float *)out->data)[i] = -((float *)input->data)[i];
  }
}
