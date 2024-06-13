
#include "nn_fill.h"


void NN_fill_F32(Tensor *tensor, float value) {
  assert(tensor->dtype == DTYPE_F32);
  
  NN__fill_F32(tensor->size, (float *)tensor->data, value);
}

void NN_fill_I32(Tensor *tensor, int32_t value) {
  assert(tensor->dtype == DTYPE_I32);
  
  NN__fill_I32(tensor->size, (int32_t *)tensor->data, value);
}

void NN_fill_I8(Tensor *tensor, int8_t value) {
  assert(tensor->dtype == DTYPE_I8);
  
  NN__fill_I8(tensor->size, (int8_t *)tensor->data, value);
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
