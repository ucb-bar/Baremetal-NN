
#include "nn_sum.h"


void NN_sum(Tensor *out, Tensor *tensor) {
  assert(out->dtype == tensor->dtype);

  switch (tensor->dtype) {
    case DTYPE_F32:
      NN__sum_F32(tensor->size, (float *)out->data, (float *)tensor->data);
      break;
    
    default:
      printf("[ERROR] Unsupported operation of tensor with dtype sum(%s)\n", 
        NN_getDataTypeName(tensor->dtype)
      );
  }
}
