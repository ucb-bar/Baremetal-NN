
#include "linear.h"


void NN_linear(Tensor *y, const Tensor *x, const Tensor *w, const Tensor *b) {
  assert(y->ndim == 1 || y->ndim == 2);
  assert(x->ndim == 1 || x->ndim == 2);
  assert(w->ndim == 2);
  assert(b->ndim == 1 || b->ndim == 2);
  assert((y->ndim == 1 && (y->shape[0] == w->shape[0])) || (y->ndim == 2 && y->shape[1] == w->shape[0]));
  assert((x->ndim == 1 && (x->shape[0] == w->shape[1])) || (x->ndim == 2 && x->shape[1] == w->shape[1]));
  assert((b->ndim == 1 && (b->shape[0] == w->shape[0])) || (b->ndim == 2 && b->shape[1] == w->shape[0]));

  size_t n_batch = y->ndim == 1 ? 1 : y->shape[0];
  size_t in_features = w->shape[1];
  size_t out_features = w->shape[0];

  NN_addmm_t_f32(n_batch, out_features, in_features, y->data, x->data, w->data, b->data);
}
