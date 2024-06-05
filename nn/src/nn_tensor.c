
#include "nn_tensor.h"


void NN_initTensor(Tensor *t, size_t ndim, size_t *shape, DataType dtype, void *data) {
  t->ndim = ndim;
  t->dtype = dtype;

  // set shape
  for (size_t i = 0; i < ndim; i += 1) {
    t->shape[i] = shape[i];
  }
  for (size_t i = ndim; i < MAX_DIMS; i += 1) {
    t->shape[i] = 0;
  }

  // set strides
  t->strides[ndim-1] = NN_sizeof(dtype);
  for (size_t i = 0; i < ndim-1; i += 1) {
    t->strides[ndim-i-2] = t->strides[ndim-i-1] * t->shape[ndim-i-1];
  }
  
  // calculate size (number of elements)
  t->size = 1;
  for (size_t i = 0; i < ndim; i += 1) {
    t->size *= t->shape[i];
  }
  
  if (data == NULL) {
    t->data = malloc(NN_sizeof(dtype) * t->size);
  } else {
    t->data = data;
  }
}


Tensor *NN_tensor(size_t ndim, size_t *shape, DataType dtype, void *data) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  NN_initTensor(t, ndim, shape, dtype, data);
  return t;
}

Tensor *NN_zeros(size_t ndim, size_t *shape, DataType dtype) {
  Tensor *t = NN_tensor(ndim, shape, dtype, NULL);

  switch (dtype) {
    case DTYPE_I8:
      for (size_t i = 0; i<t->size; i+=1) {
        ((int8_t *)t->data)[i] = 0;
      }
      break;
    case DTYPE_I32:
      for (size_t i = 0; i<t->size; i+=1) {
        ((int32_t *)t->data)[i] = 0;
      }
      break;
    case DTYPE_F32:
      for (size_t i = 0; i<t->size; i+=1) {
        ((float *)t->data)[i] = 0;
      }
      break;
    default:
      printf("[WARNING] Unsupported data type: %d\n", dtype);
  }

  return t;
}

Tensor *NN_ones(size_t ndim, size_t *shape, DataType dtype) {
  Tensor *t = NN_tensor(ndim, shape, dtype, NULL);

  switch (dtype) {
    case DTYPE_I8:
      for (size_t i = 0; i<t->size; i+=1) {
        ((int8_t *)t->data)[i] = 1;
      }
      break;
    case DTYPE_I32:
      for (size_t i = 0; i<t->size; i+=1) {
        ((int32_t *)t->data)[i] = 1;
      }
      break;
    case DTYPE_F32:
      for (size_t i = 0; i<t->size; i+=1) {
        ((float *)t->data)[i] = 1;
      }
      break;
    default:
      printf("[WARNING] Unsupported data type: %d\n", dtype);
  }

  return t;
}

void NN_asType(Tensor *t, DataType dtype) {
  if (t->dtype == dtype) {
    return;
  }
  if (t->dtype == DTYPE_I32 && dtype == DTYPE_F32) {
    for (size_t i = 0; i<t->size; i+=1) {
      ((float *)t->data)[i] = (float)((int32_t *)t->data)[i];
    }
    t->dtype = DTYPE_F32;
    return;
  }
  if (t->dtype == DTYPE_I32 && dtype == DTYPE_I8) {
    for (size_t i = 0; i<t->size; i+=1) {
      ((int8_t *)t->data)[i] = (int8_t)((int32_t *)t->data)[i];
    }
    t->dtype = DTYPE_I8;
    return;
  }

  if (t->dtype == DTYPE_F32 && dtype == DTYPE_I32) {
    for (size_t i = 0; i<t->size; i+=1) {
      ((int32_t *)t->data)[i] = (int32_t)((float *)t->data)[i];
    }
    t->dtype = DTYPE_I32;
    return;
  }

  printf("[ERROR] Cannot convert data type from %s to %s\n", NN_getDataTypeName(t->dtype), NN_getDataTypeName(dtype));
}
