
#include "nn_tensor.h"


void NN_initTensor(Tensor *tensor, const size_t ndim, const size_t *shape, DataType dtype, void *data) {
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
  NN_initTensor(t, ndim, shape, dtype, data);
  return t;
}

void NN_asType(Tensor *out, Tensor *in) {
  if (out->dtype == in->dtype) {
    NN_copy(out, in);
    return;
  }

  switch (in->dtype) {
    case DTYPE_I8:
      switch (out->dtype) {
        case DTYPE_I32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((int32_t *)out->data)[i] = (int32_t)((int8_t *)in->data)[i];
          }
          return;
        case DTYPE_F32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((float *)out->data)[i] = (float)((int8_t *)in->data)[i];
          }
          return;
      }
      break;
  
    case DTYPE_I32:
      switch (out->dtype) {
        case DTYPE_I8:
          for (size_t i = 0; i < in->size; i += 1) {
            ((int8_t *)out->data)[i] = (int8_t)((int32_t *)in->data)[i];
          }
          return;
        case DTYPE_F32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((float *)out->data)[i] = (float)((int32_t *)in->data)[i];
          }
          return;
      }
      break;
    
    case DTYPE_F16:
      switch (out->dtype) {
        case DTYPE_F32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((float *)out->data)[i] = NN_halfToFloat(((float16_t *)in->data)[i]);
          }
          return;
      }
      break;
    
    case DTYPE_F32:
      switch (out->dtype) {
        case DTYPE_I32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((int32_t *)out->data)[i] = (int32_t)((float *)in->data)[i];
          }
          return;
        case DTYPE_F16:
          for (size_t i = 0; i < in->size; i += 1) {
            ((float16_t *)out->data)[i] = NN_floatToHalf(((float *)in->data)[i]);
          }
          return;
      }
      break;
  }
  printf("[ERROR] Cannot convert data type from %s to %s\n", NN_getDataTypeName(in->dtype), NN_getDataTypeName(out->dtype));
}
