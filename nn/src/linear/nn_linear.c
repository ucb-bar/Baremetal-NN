
#include "nn_linear.h"

void NN_Linear_F16(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  NN_matmulT_F16(y, x, w);

  if (b != NULL) {
    NN_add_F16(y, y, b);
  }
}

void NN_Linear_F32(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  NN_matmulT_F32(y, x, w);

  if (b != NULL) {
    NN_add_F32(y, y, b);
  }
}
