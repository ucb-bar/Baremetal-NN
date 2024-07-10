
#include "nn_tensor_creation.h"


void NN_init_tensor(Tensor *tensor, const size_t ndim, const size_t *shape, DataType dtype, void *data) {
  tensor->dtype = dtype;
  tensor->ndim = ndim;

  // set shape
  memcpy(tensor->shape, shape, ndim * sizeof(size_t));
  memset(tensor->shape + ndim, 0, (MAX_DIMS - ndim) * sizeof(size_t));

  // calculate size (number of elements)
  tensor->size = 1;
  for (size_t i = 0; i < ndim; i += 1) {
    tensor->size *= shape[i];
  }
  
  if (data != NULL) {
    tensor->data = data;
    return;
  }

  // if this is a scalar tensor
  if (tensor->ndim == 0) {
    tensor->data = malloc(NN_sizeof(dtype));
    return;
  }
  
  tensor->data = malloc(NN_sizeof(dtype) * tensor->size);
}

Tensor *NN_tensor(size_t ndim, const size_t *shape, DataType dtype, void *data) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  NN_init_tensor(t, ndim, shape, dtype, data);
  return t;
}
