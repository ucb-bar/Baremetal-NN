
#include "nn_softmax.h"


void NN_softmax(Tensor *out, Tensor *tensor, size_t dim) {
  assert(out->dtype == tensor->dtype);

  switch (tensor->dtype) {
    case DTYPE_F32:
      if (dim == 0) {
        for (size_t i = 0; i < tensor->shape[0]; i += 1) {
          float *x = (float *)tensor->data + i * tensor->shape[1];
          float *y = (float *)out->data + i * out->shape[1];
          NN__softmax_f32(tensor->shape[1], y, x, 1);
        }
        return;
      }
      if (dim == 1) {
        for (size_t i = 0; i < tensor->shape[1]; i += 1) {
          float *x = (float *)tensor->data + i;
          float *y = (float *)out->data + i;
          NN__softmax_f32(tensor->shape[0], y, x, tensor->shape[1]);
        }
        return;
      }

      // NN__softmax_f32(tensor->size, (float *)out->data, (float *)tensor->data);
      break;
    
    default:
      printf("[ERROR] Unsupported operation of tensor with dtype %s = softmax(%s)\n", 
        NN_get_datatype_name(out->dtype), NN_get_datatype_name(tensor->dtype)
      );
  }
}
