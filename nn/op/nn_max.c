
#include "nn_max.h"


void NN_max(Tensor *scalar, Tensor *tensor) {
  assert(scalar->dtype == tensor->dtype);

  switch (tensor->dtype) {
    case DTYPE_F32:
      NN__max_f32(tensor->size, (float *)scalar->data, (float *)tensor->data, 1);
      break;
    
    default:
      printf("[ERROR] Unsupported operation of tensor with dtype max(%s)\n", 
        NN_get_datatype_name(tensor->dtype)
      );
  }
}
