
#include "nn_copy.h"

void NN_copy(Tensor *dst, Tensor *src) {
  assert(dst->ndim == src->ndim);
  assert(dst->dtype == src->dtype);
  assert(dst->size == src->size);
  
  memcpy(dst->data, src->data, dst->size * NN_sizeof(dst->dtype));
}
