
#include "silu.h"


void NN_silu(Tensor *y, const Tensor *x) {
  assert(y->ndim == x->ndim);
  assert(y->dtype == x->dtype);
  assert(y->size == x->size);

  switch (y->dtype) {
    case DTYPE_F32:
      for (size_t i = 0; i < y->size; i++) {
        float x_i = ((float *)x->data)[i];
        ((float *)y->data)[i] = x_i / (1.0f + expf(-x_i));
      }
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = SiLU(%s)\n", 
    NN_get_datatype_name(y->dtype), NN_get_datatype_name(x->dtype)
  );
}

void NN_silu_inplace(Tensor *x) {
  NN_silu(x, x);
}
