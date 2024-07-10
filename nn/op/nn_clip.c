
#include "nn_clip.h"


void NN_clip(Tensor *y, Tensor *x, float min, float max) {
  assert(y->ndim == x->ndim);
  assert(y->dtype == x->dtype);
  assert(y->size == x->size);

  switch (y->dtype) {
    case DTYPE_F32:
      NN__maximum1_f32(y->size, (float *)y->data, 1, (float *)x->data, 1, min);
      NN__minimum1_f32(y->size, (float *)y->data, 1, (float *)y->data, 1, max);
      return;

    default:
      break;
  }

  printf("[ERROR] Unsupported operation for tensor with dtype %s = clip(%s, float, float)\n", 
    NN_get_datatype_name(y->dtype), NN_get_datatype_name(x->dtype)
  );
}

void NN_clip_inplace(Tensor *x, float min, float max) {
  NN_clip(x, x, min, max);
}
