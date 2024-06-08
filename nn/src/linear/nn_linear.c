
#include "nn_linear.h"

void NN_linear_F32(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  NN_transpose_F32(w, w);
  NN_matmul_F32(y, x, w);
  NN_add_F32(y, y, b);
}
