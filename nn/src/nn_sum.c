
#include "nn_sum.h"


float NN_sum_F32(Tensor *tensor) {
  assert(tensor->dtype == DTYPE_F32);
  
  float sum;

  NN__sum_F32(tensor->size, &sum, (float *)tensor->data);

  return sum;
}
