
#include "nn_sum.h"


float NN_sum_F32(Tensor *tensor) {
  assert(tensor->dtype == DTYPE_F32);
  
  float sum = 0;

  for (size_t i = 0; i < tensor->size; i += 1) {
    sum += ((float *)tensor->data)[i];
  }

  return sum;
}
