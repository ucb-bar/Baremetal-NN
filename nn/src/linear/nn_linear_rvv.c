
#include "nn_linear.h"
#include "riscv_vector.h"

void NN_linear_F32_RVV(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  Tensor *wt = NN_tensor(2, w->shape, DTYPE_F32, w->data);
  NN_transpose_F32(wt, w);
  NN_matmul_F32_RVV(y, x, wt);
  NN_add_F32_RVV(y, y, b);
}
