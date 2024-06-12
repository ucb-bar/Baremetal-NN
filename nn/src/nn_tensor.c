
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

Tensor *NN_zeros(size_t ndim, const size_t *shape, DataType dtype) {
  Tensor *t = NN_tensor(ndim, shape, dtype, NULL);

  switch (dtype) {
    case DTYPE_I8:
      NN_fill_I8(t, 0);
      break;
    case DTYPE_I32:
      NN_fill_I32(t, 0);
      break;
    case DTYPE_F16:
      NN_fill_F16(t, 0);
      break;
    case DTYPE_F32:
      NN_fill_F32(t, 0);
      break;
    default:
      printf("[WARNING] Unsupported data type: %d\n", dtype);
  }

  return t;
}

Tensor *NN_ones(size_t ndim, const size_t *shape, DataType dtype) {
  Tensor *t = NN_tensor(ndim, shape, dtype, NULL);

  switch (dtype) {
    case DTYPE_I8:
      NN_fill_I8(t, 1);
      break;
    case DTYPE_I32:
      NN_fill_I32(t, 1);
      break;
    case DTYPE_F16:
      NN_fill_F16(t, 1);
      break;
    case DTYPE_F32:
      NN_fill_F32(t, 1);
      break;
    default:
      printf("[WARNING] Unsupported data type: %d\n", dtype);
  }

  return t;
}

Tensor *NN_rand(size_t ndim, const size_t *shape, DataType dtype) {
  Tensor *t = NN_tensor(ndim, shape, dtype, NULL);

  switch (dtype) {
    case DTYPE_I8:
      for (size_t i = 0; i < t->size; i += 1) {
        ((int8_t *)t->data)[i] = rand() % 256;
      }
      break;
    case DTYPE_I32:
      for (size_t i = 0; i < t->size; i += 1) {
        ((int32_t *)t->data)[i] = rand();
      }
      break;
    case DTYPE_F16:
      for (size_t i = 0; i < t->size; i += 1) {
        ((float16_t *)t->data)[i] = NN_floatToHalf((float)rand() / RAND_MAX);
      }
      break;
    case DTYPE_F32:
      for (size_t i = 0; i < t->size; i += 1) {
        ((float *)t->data)[i] = (float)rand() / RAND_MAX;
      }
      break;
    default:
      printf("[WARNING] Unsupported data type: %d\n", dtype);
  }

  return t;
}

void NN_asType(Tensor *out, Tensor *in) {
  assert(out->size == in->size);

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
