
#include "nn_abs.h"


void NN_abs(Tensor *out, Tensor *in) {
  assert(out->ndim == in->ndim);
  assert(out->dtype == in->dtype);
  assert(out->size == in->size);
  
  switch (out->dtype) {
    case DTYPE_F32:
      NN_abs_F32(out, in);
      return;

    default:
  }
  printf("[ERROR] Unsupported operation of tensor with dtype %s = |%s|\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(in->dtype)
  );
}
