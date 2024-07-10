
#include "nn_linear.h"


void NN_linear(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  NN_mm_t(y, x, w);

  if (b != NULL) {
    NN_add(y, y, b);
  }
}
