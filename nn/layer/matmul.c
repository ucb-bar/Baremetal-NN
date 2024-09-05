
#include "matmul.h"


// for mv() operation, NN_matmul(mat, vec) is equivalent to NN_matmul_t(vec, mat)

void NN_matmul(Tensor *out, const Tensor *a, const Tensor *b) {
  if (a->ndim == 2 && b->ndim == 1) {
    NN_mv(out, a, b);
    return;
  }
  if (a->ndim == 2 && b->ndim == 2) {
    NN_mm(out, a, b);
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

void NN_matmul_t(Tensor *out, const Tensor *a, const Tensor *b) {
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

