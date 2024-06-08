
#include "nn_max.h"


float NN_max_F32(Tensor *tensor) {
  assert(tensor->dtype == DTYPE_F32);
  
  float max = -FLT_MAX;
  uint8_t *ptr = tensor->data;
  
  switch (tensor->ndim) {
    case 1:
      for (size_t i = 0; i < tensor->shape[0]; i += 1) {
        float val = *((float *)ptr);
        max = val > max ? val : max;
        ptr += tensor->strides[0];
      }
      break;
    case 2:
      for (size_t i = 0; i < tensor->shape[0]; i += 1) {
        for (size_t j = 0; j < tensor->shape[1]; j += 1) {
          float val = *((float *)ptr);
          max = val > max ? val : max;
          ptr += tensor->strides[1];
        }
        ptr -= tensor->strides[1] * tensor->shape[1];
        ptr += tensor->strides[0];
      }
      break;
  }
  return max;
}
