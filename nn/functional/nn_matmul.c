
#include "nn_matmul.h"


void NN_matmul(Tensor *out, Tensor *a, Tensor *b) {
  if (a->ndim == 2 && b->ndim == 2) {
    NN_mm(out, a, b);
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

void NN_matmul_t(Tensor *out, Tensor *a, Tensor *b) {
  if (a->ndim == 1 && b->ndim == 2) {
    NN_mv(out, b, a);
    return;
  }
  if (a->ndim == 2 && b->ndim == 2) {
    NN_mm_t(out, a, b);
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

