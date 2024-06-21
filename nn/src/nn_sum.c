
#include "nn_sum.h"


void NN_sum(Tensor *out, Tensor *tensor) {
  switch (tensor->dtype) {
    case DTYPE_I16:
      switch (out->dtype) {
        case DTYPE_I32:
          NN__sum_I16_to_I32(tensor->size, (int32_t *)out->data, (int16_t *)tensor->data);
          return;
      }
      break;

    case DTYPE_I32:
      switch (out->dtype) {
        case DTYPE_I32:
          NN__sum_I32(tensor->size, (int32_t *)out->data, (int32_t *)tensor->data);
          return;
      }
      break;
    
    case DTYPE_F32:
      NN__sum_F32(tensor->size, (float *)out->data, (float *)tensor->data);
      return;
    
    default:
      break;
  }

  printf("[ERROR] Unsupported operation of tensor with dtype %s = sum(%s)\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(tensor->dtype)
  );
}
