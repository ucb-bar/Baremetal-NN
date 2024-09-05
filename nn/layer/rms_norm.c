
#include "rms_norm.h"

void NN_rms_norm(Tensor *y, const Tensor *x, const Tensor *w, float eps) {
  assert(y->ndim == 1);
  assert(x->ndim == 1);
  assert(w->ndim == 1);
  assert(y->size == x->size);
  assert(y->size == w->size);

  switch (x->dtype) {
    case DTYPE_F32:
      NN_rms_norm_f32(y->shape[0],
        (float *)y->data, 1,
        (float *)x->data, 1,
        (float *)w->data, 1,
        eps
      );
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = RMSNorm(%s)\n", 
    NN_get_datatype_name(y->dtype), NN_get_datatype_name(x->dtype)
  );
}
