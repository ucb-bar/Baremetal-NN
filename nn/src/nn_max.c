
#include "nn_max.h"


float NN_max(Tensor *tensor) {
  float max;

  switch (tensor->dtype) {
    case DTYPE_F32:
      NN__max_F32(tensor->size, &max, (float *)tensor->data);
      break;
    
    default:
      printf("[ERROR] Unsupported operation of tensor with dtype max(%s)\n", 
        NN_getDataTypeName(tensor->dtype)
      );
  }
  
  return max;
}
