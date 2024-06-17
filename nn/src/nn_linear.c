
#include "nn_linear.h"


void NN_Linear(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  NN_matmulT(y, x, w);

  if (b != NULL) {
    NN_add(y, y, b);
  }
}
