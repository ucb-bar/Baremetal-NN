
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

void NN_linear_I32(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  assert(x->shape[1] == w->shape[0]);
  assert(y->shape[0] == x->shape[0]);
  assert(y->shape[1] == w->shape[1]);
  assert(b->shape[0] == w->shape[1]);
  assert(x->dtype == DTYPE_I32);
  assert(w->dtype == DTYPE_I32);
  assert(b->dtype == DTYPE_I32);
  assert(y->dtype == DTYPE_I32);

  NN_matmul_I32(y, x, w);
  NN_add_I32(y, y, b);
}

void NN_linear_I8_I8_I8_I32(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  assert(x->dtype == DTYPE_I8);
  assert(w->dtype == DTYPE_I8);
  assert(b->dtype == DTYPE_I8);
  assert(y->dtype == DTYPE_I32);

  NN_matmul_I8_I8_I32(y, x, w);
  NN_add_I32_I8_I32(y, y, b);
}
