
#include "nn_max.h"


float NN_max_F32(Tensor *tensor) {
  assert(tensor->dtype == DTYPE_F32);
  
  float max = -FLT_MAX;
  
  for (size_t i = 0; i < tensor->size; i += 1) {
    float val = ((float *)tensor->data)[i];
    if (val > max) {
      max = val;
    }
  }
  
  return max;
}
