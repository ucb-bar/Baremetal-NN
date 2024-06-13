
#include "nn_min.h"


float NN_min(Tensor *tensor) {
  float min;

  switch (tensor->dtype) {
    case DTYPE_F32:
      NN__min_F32(tensor->size, &min, (float *)tensor->data);
      break;
    
    default:
      printf("[ERROR] Unsupported operation of tensor with dtype min(%s)\n", 
        NN_getDataTypeName(tensor->dtype)
      );
  }
  
  return min;
}
