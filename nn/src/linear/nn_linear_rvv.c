
#include "nn_linear.h"
#include "riscv_vector.h"

void NN_Linear_F32_RVV(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  NN_matmulT_F32_RVV(y, x, w);

  if (b != NULL) {
    NN_add_F32_RVV(y, y, b);
  }
}
