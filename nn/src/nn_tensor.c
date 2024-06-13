
#include "nn_tensor.h"


void NN_initTensor(Tensor *tensor, size_t ndim, const size_t *shape, DataType dtype, void *data) {
  tensor->ndim = ndim;
  tensor->dtype = dtype;

  // set shape
  for (size_t i = 0; i < ndim; i += 1) {
    tensor->shape[i] = shape[i];
  }
  for (size_t i = ndim; i < MAX_DIMS; i += 1) {
    tensor->shape[i] = 0;
  }

  // calculate size (number of elements)
  tensor->size = 1;
  for (size_t i = 0; i < ndim; i += 1) {
    tensor->size *= tensor->shape[i];
  }
  
  if (data == NULL) {
    tensor->data = malloc(NN_sizeof(dtype) * tensor->size);
  } else {
    tensor->data = data;
  }
}

Tensor *NN_tensor(size_t ndim, const size_t *shape, DataType dtype, void *data) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  NN_initTensor(t, ndim, shape, dtype, data);
  return t;
}

void NN_asType(Tensor *t, DataType dtype) {
  if (t->dtype == dtype) {
    return;
  }
  if (t->dtype == DTYPE_I32 && dtype == DTYPE_F32) {
    for (size_t i = 0; i < t->size; i += 1) {
      ((float *)t->data)[i] = (float)((int32_t *)t->data)[i];
    }
    t->dtype = DTYPE_F32;
    return;
  }
  if (t->dtype == DTYPE_I32 && dtype == DTYPE_I8) {
    for (size_t i = 0; i < t->size; i += 1) {
      ((int8_t *)t->data)[i] = (int8_t)((int32_t *)t->data)[i];
    }
    t->dtype = DTYPE_I8;
    return;
  }

  if (t->dtype == DTYPE_F32 && dtype == DTYPE_I32) {
    for (size_t i = 0; i < t->size; i += 1) {
      ((int32_t *)t->data)[i] = (int32_t)((float *)t->data)[i];
    }
    t->dtype = DTYPE_I32;
    return;
  }

  printf("[ERROR] Cannot convert data type from %s to %s\n", NN_getDataTypeName(t->dtype), NN_getDataTypeName(dtype));
}
