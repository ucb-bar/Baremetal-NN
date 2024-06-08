
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
  if (a->dtype == DTYPE_F32) {
    NN_transpose_F32(out, a);
    return;
  }
  printf("Unsupported data type %s\n", NN_getDataTypeName(a->dtype));
}

void NN_transpose_F32(Tensor *out, Tensor *a) {
  // currently only support 2D matrix transpose
  assert(a->ndim == 2);
  assert(a->dtype == DTYPE_F32);
  assert(out->ndim == 2);
  assert(out->dtype == DTYPE_F32);
  
  for (size_t i = 0; i < a->shape[0]; i += 1) {
    for (size_t j = 0; j < a->shape[1]; j += 1) {
      ((float *)out->data)[j * out->shape[1] + i] = ((float *)a->data)[i * a->shape[1] + j];
    }
  }
}

