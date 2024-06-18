
#include "nn_abs.h"


void NN_abs(Tensor *out, Tensor *in) {
  assert(out->ndim == in->ndim);
  assert(out->dtype == in->dtype);
  assert(out->size == in->size);
  
  switch (out->dtype) {
    case DTYPE_F16:
      NN__abs_F16(out->size, (float16_t *)out->data, (float16_t *)in->data);
      return;
    case DTYPE_F32:
      NN__abs_F32(out->size, (float *)out->data, (float *)in->data);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = |%s|\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(in->dtype)
  );
}
