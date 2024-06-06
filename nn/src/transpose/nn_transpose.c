
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
  
  out->dtype = DTYPE_F32;

  size_t shape0 = a->shape[0];
  out->shape[0] = a->shape[1];
  out->shape[1] = shape0;

  size_t strides0 = a->strides[0];
  out->strides[0] = a->strides[1];
  out->strides[1] = strides0;
}

