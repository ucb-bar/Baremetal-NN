
#include "nn_clip.h"


void NN_clip(Tensor *y, Tensor *x, float min, float max) {
  assert(y->ndim == x->ndim);
  assert(y->dtype == x->dtype);
  assert(y->size == x->size);

  switch (y->dtype) {
    case DTYPE_F32:
      NN__maximum1_F32(y->size, (float *)y->data, (float *)x->data, min);
      NN__minimum1_F32(y->size, (float *)y->data, (float *)y->data, max);
      return;

    default:
  }

  printf("[ERROR] Unsupported operation for tensor with dtype %s = clip(%s, float, float)\n", 
    NN_getDataTypeName(y->dtype), NN_getDataTypeName(x->dtype)
  );
}

void NN_clipInplace(Tensor *x, float min, float max) {
  NN_clip(x, x, min, max);
}
