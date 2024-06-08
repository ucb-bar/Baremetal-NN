
#include "nn_matmul.h"

void NN_matmul_I8_I8_I32_EAGLEX(Tensor *out, Tensor *a, Tensor *b) {
  // TODO: port to here
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(b->shape[0] == a->shape[1]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[1]);
}

