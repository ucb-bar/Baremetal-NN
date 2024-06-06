
#include "nn_min.h"


float NN_min_F32(Tensor *t) {
  assert(t->dtype == DTYPE_F32);
  
  float min = FLT_MAX;
  float *t_data = (float *)t->data;

  for (size_t i = 0; i < t->shape[0]; i += 1) {
    for (size_t j = 0; j < t->shape[1]; j += 1) {
      if (t_data[i * t->shape[1] + j] < min) {
        min = t_data[i * t->shape[1] + j];
      }
    }
  }
  
  return min;
}