
#include "nn_linear.h"

void NN_linear_F32(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  assert(x->shape[1] == w->shape[0]);
  assert(y->shape[0] == x->shape[0]);
  assert(y->shape[1] == w->shape[1]);
  // assert(b->shape[1] == w->shape[1]);
  assert(x->dtype == DTYPE_F32);
  assert(w->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(y->dtype == DTYPE_F32);

  NN_matmul_F32(y, x, w);
  NN_add_F32(y, y, b);
}
