
#include "nn_sum.h"


float NN_sum_F32(Tensor *tensor) {
  assert(tensor->dtype == DTYPE_F32);
  
  uint8_t *ptr = tensor->data;
  float sum = 0;
  
  switch (tensor->ndim) {
    case 0:
      return sum;
    
    case 1:
      for (size_t i = 0; i < tensor->shape[0]; i += 1) {
        sum += *((float *)ptr);
        ptr += tensor->strides[0];
      }
      return sum;
  
    case 2:
      for (size_t i = 0; i < tensor->shape[0]; i += 1) {
        for (size_t j = 0; j < tensor->shape[1]; j += 1) {
          sum += *((float *)ptr);
          ptr += tensor->strides[1];
        }
        // reset to the beginning of the row
        ptr -= tensor->strides[1] * tensor->shape[1];
        ptr += tensor->strides[0];
      }
      return sum;
  }

  return sum;
}
