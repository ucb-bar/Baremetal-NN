
#include "nn_fill.h"


void NN_fill(Tensor *tensor, float value) {
  switch (tensor->dtype) {
    case DTYPE_I8:
      NN__fill_I8(tensor->size, (int8_t *)tensor->data, (int8_t)value);
      return;
    case DTYPE_I32:
      NN__fill_I32(tensor->size, (int32_t *)tensor->data, (int32_t)value);
      return;
    case DTYPE_F16:
      NN__fill_F16(tensor->size, (float16_t *)tensor->data, NN_floatToHalf(value));
      return;
    case DTYPE_F32:
      NN__fill_F32(tensor->size, (float *)tensor->data, value);
      return;
    default:
      printf("[ERROR] Unsupported operation fill to tensor with dtype: %d\n", tensor->dtype);
  }
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
