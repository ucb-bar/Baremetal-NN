
#include "nn_min.h"


void NN_min(Tensor *scalar, const Tensor *tensor) {
  assert(scalar->dtype == tensor->dtype);

  switch (tensor->dtype) {
    case DTYPE_F32:
      NN_min_f32(tensor->size, (float *)scalar->data, (float *)tensor->data, 1);
      break;
    
    default:
      printf("[ERROR] Unsupported operation of tensor with dtype min(%s)\n", 
        NN_get_datatype_name(tensor->dtype)
      );
  }
}
