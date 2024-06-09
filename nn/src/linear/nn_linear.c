
#include "nn_linear.h"

void NN_Linear_F32(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  NN_matmulT_F32(y, x, w);
  NN_add_F32(y, y, b);
}
