
#include "nn_transpose.h"


/**
 * Transpose a 2D tensor
 * 
 * @warning this is not an in-place operation, the output tensor should be different from the input tensor
 * 
 * @param out: output tensor of shape (n, m)
 * @param a: input tensor of shape (m, n)
 */
void NN_transpose(Tensor *out, Tensor *a) {
  assert(a->ndim == 2);
  assert(out->ndim == a->ndim);
  assert(out->dtype == a->dtype);
  assert(out->shape[0] == a->shape[1]);
  assert(out->shape[1] == a->shape[0]);

  if (a->dtype == DTYPE_F32) {
    NN__transpose_F32(a->shape[0], a->shape[1], (float *)out->data, (float *)a->data);
    return;
  }

  printf("[ERROR] Unsupported operation of tensor with dtype %s = %s.T\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype)
  );
}
