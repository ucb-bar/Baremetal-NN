
#include "nn_min.h"


float NN_min_F32(Tensor *tensor) {
  assert(tensor->dtype == DTYPE_F32);
  
  float min = FLT_MAX;
  
  for (size_t i = 0; i < tensor->size; i += 1) {
    float val = ((float *)tensor->data)[i];
    if (val < min) {
      min = val;
    }
  }
  
  return min;
}
