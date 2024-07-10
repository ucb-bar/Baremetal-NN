
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

Tensor *NN_zeros(size_t ndim, const size_t *shape, DataType dtype) {
  Tensor *t = NN_tensor(ndim, shape, dtype, NULL);

  NN_fill(t, 0);

  return t;
}

Tensor *NN_ones(size_t ndim, const size_t *shape, DataType dtype) {
  Tensor *t = NN_tensor(ndim, shape, dtype, NULL);

  NN_fill(t, 1);

  return t;
}

Tensor *NN_rand(size_t ndim, const size_t *shape, DataType dtype) {
  Tensor *t = NN_tensor(ndim, shape, dtype, NULL);

  switch (dtype) {
    case DTYPE_U8:
      for (size_t i = 0; i < t->size; i += 1) {
        ((uint8_t *)t->data)[i] = rand() % 0x100;
      }
      break;
    case DTYPE_I8:
      for (size_t i = 0; i < t->size; i += 1) {
        ((int8_t *)t->data)[i] = rand() % 0x100;
      }
      break;
    case DTYPE_U16:
      for (size_t i = 0; i < t->size; i += 1) {
        ((uint16_t *)t->data)[i] = rand() % 0x10000;
      }
      break;
    case DTYPE_I32:
      for (size_t i = 0; i < t->size; i += 1) {
        ((int32_t *)t->data)[i] = rand();
      }
      break;
    case DTYPE_F32:
      for (size_t i = 0; i < t->size; i += 1) {
        ((float *)t->data)[i] = (float)rand() / RAND_MAX;
      }
      break;
    default:
      printf("[ERROR] Unsupported data type: %d\n", dtype);
  }

  return t;
}
