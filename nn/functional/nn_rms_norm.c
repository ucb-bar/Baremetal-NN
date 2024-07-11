
#include "nn_rms_norm.h"

void NN_rms_norm(Tensor *y, Tensor *x, Tensor *w) {
  switch (x->dtype) {
    case DTYPE_F32:
      NN__rms_norm_f32(x->shape[0] * x->shape[1],
        (float *)y->data, 1,
        (float *)x->data, 1,
        (float *)w->data, 1
      );
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = RMSNorm(%s)\n", 
    NN_get_datatype_name(y->dtype), NN_get_datatype_name(x->dtype)
  );
}
