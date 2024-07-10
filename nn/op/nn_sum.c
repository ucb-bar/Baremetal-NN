
#include "nn_sum.h"


void NN_sum(Tensor *out, Tensor *tensor) {
  switch (tensor->dtype) {
    case DTYPE_I16:
      switch (out->dtype) {
        case DTYPE_I32:
          NN__sum_i16_to_i32(tensor->size, (int32_t *)out->data, (int16_t *)tensor->data, 1);
          return;
      }
      break;

    case DTYPE_I32:
      switch (out->dtype) {
        case DTYPE_I32:
          NN__sum_i32(tensor->size, (int32_t *)out->data, (int32_t *)tensor->data, 1);
          return;
      }
      break;
    
    case DTYPE_F32:
      NN__sum_f32(tensor->size, (float *)out->data, (float *)tensor->data, 1);
      return;
    
    default:
      break;
  }

  printf("[ERROR] Unsupported operation of tensor with dtype %s = sum(%s)\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(tensor->dtype)
  );
}
