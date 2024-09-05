
#include "linear.h"


void NN_linear(Tensor *y, const Tensor *x, const Tensor *w, const Tensor *b) {
  NN_matmul_t(y, x, w);

  if (b != NULL) {
    NN_add(y, y, b);
  }
}
