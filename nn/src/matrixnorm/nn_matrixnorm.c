
#include "nn_matrixnorm.h"


float NN_matrixNorm_F32(Tensor *tensor) {
  assert(tensor->ndim == 2);
  assert(tensor->dtype == DTYPE_F32);

  float sum = 0;
  for (int i = 0; i < tensor->shape[0]; i += 1) {
    for (int j = 0; j < tensor->shape[1]; j += 1) {
      sum += pow(((float *)tensor->data)[i * tensor->shape[1] + j], 2);
    }
  }
  return sqrt(sum);
}
