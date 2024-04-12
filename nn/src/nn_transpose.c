
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
  assert(out->shape[0] == a->shape[1]);
  assert(out->shape[1] == a->shape[0]);
  assert(a->ndim == 2);
  assert(out->dtype == a->dtype);
  assert(out->data != a->data);

  if (a->dtype == DTYPE_F32) {
    NN_transpose_F32(out, a);
    return;
  }
  if (a->dtype == DTYPE_I8) {
    NN_transpose_I8(out, a);
    return;
  }
  if (a->dtype == DTYPE_I32) {
    NN_transpose_I32(out, a);
    return;
  }
  printf("Unsupported data type %s\n", NN_getDataTypeName(a->dtype));
}

void NN_transpose_I8(Tensor *out, Tensor *a) {
  assert(out->shape[0] == a->shape[1]);
  assert(out->shape[1] == a->shape[0]);
  assert(a->ndim == 2);
  assert(out->dtype == a->dtype);

  for (size_t i = 0; i<a->shape[0]; i+=1) {
    for (size_t j = 0; j<a->shape[1]; j+=1) {
      ((int8_t *)out->data)[i*a->shape[1]+j] = ((int8_t *)a->data)[j*a->shape[0]+i];
    }
  }
}

void NN_transpose_I32(Tensor *out, Tensor *a) {
  assert(out->shape[0] == a->shape[1]);
  assert(out->shape[1] == a->shape[0]);
  assert(a->ndim == 2);
  assert(out->dtype == a->dtype);

  for (size_t i = 0; i<a->shape[0]; i+=1) {
    for (size_t j = 0; j<a->shape[1]; j+=1) {
      ((int32_t *)out->data)[i*a->shape[1]+j] = ((int32_t *)a->data)[j*a->shape[0]+i];
    }
  }
}

void NN_transpose_F32(Tensor *out, Tensor *a) {
  assert(out->shape[0] == a->shape[1]);
  assert(out->shape[1] == a->shape[0]);
  assert(a->ndim == 2);
  assert(out->dtype == a->dtype);

  for (size_t i = 0; i<a->shape[0]; i+=1) {
    for (size_t j = 0; j<a->shape[1]; j+=1) {
      ((float *)out->data)[i*a->shape[1]+j] = ((float *)a->data)[j*a->shape[0]+i];
    }
  }
}

