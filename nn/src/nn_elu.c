
#include "nn_elu.h"


void NN_ELU(Tensor *y, Tensor *x, float alpha) {
  assert(y->ndim == x->ndim);
  assert(y->dtype == x->dtype);
  assert(y->size == x->size);

  switch (y->dtype) {
    case DTYPE_F32:
      for (size_t i = 0; i < y->shape[0] * y->shape[1]; i += 1) {
        if (((float *)x->data)[i] > 0) {
          ((float *)y->data)[i] = ((float *)x->data)[i];
        }
        else {
          ((float *)y->data)[i] = alpha * (expf(((float *)x->data)[i]) - 1.f);
        }
      }
      // NN__elu_F32(y->size, (float *)y->data, (float *)x->data, 0.0f);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = ELU(%s)\n", 
    NN_getDataTypeName(y->dtype), NN_getDataTypeName(x->dtype)
  );
}

void NN_ELUInplace(Tensor *x, float alpha) {
  NN_ELU(x, x, alpha);
}
