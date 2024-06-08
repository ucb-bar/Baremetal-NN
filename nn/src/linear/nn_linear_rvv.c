
#include "nn_linear.h"
#include "riscv_vector.h"

void NN_linear_F32_RVV(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  NN_matmult_F32_RVV(y, x, w);
  NN_add_F32_RVV(y, y, b);
}
