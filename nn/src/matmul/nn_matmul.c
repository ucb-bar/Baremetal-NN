
#include "nn_matmul.h"

void NN_matmul(Tensor *out, Tensor *a, Tensor *b) {
  if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8 && out->dtype == DTYPE_I32) {
    NN_matmul_I8_I8_I32(out, a, b);
    return;
  }
  if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I32 && out->dtype == DTYPE_I32) {
    NN_matmul_I32(out, a, b);
    return;
  }
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
    NN_matmul_F32(out, a, b);
    return;
  }
  printf("Unsupported operation: %s @ %s -> %s\n", NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype), NN_getDataTypeName(out->dtype));
}

void NN_matmul_F32(Tensor *out, Tensor *a, Tensor *b) {
  // currently only support 2D matrix multiplication
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(a->shape[1] == b->shape[0]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[1]);

  for (size_t i = 0; i < out->shape[0]; i += 1) {
    for (size_t j = 0; j < out->shape[1]; j += 1) {
      float sum = 0;
      for (size_t k = 0; k < a->shape[1]; k += 1) {
        sum += ((float *)a->data)[i * a->shape[1] + k] * ((float *)b->data)[k * b->shape[1] + j];
      }
      ((float *)out->data)[i * out->shape[1] + j] = sum;
    }
  }
}

void NN_matmulT_F16(Tensor *out, Tensor *a, Tensor *b) {
  // currently only support 2D matrix multiplication
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_F16);
  assert(b->dtype == DTYPE_F16);
  assert(out->dtype == DTYPE_F16);
  assert(a->shape[1] == b->shape[1]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[0]);

  for (size_t i = 0; i < out->shape[0]; i += 1) {
    for (size_t j = 0; j < out->shape[1]; j += 1) {
      float16_t sum = 0;
      for (size_t k = 0; k < a->shape[1]; k += 1) {
        sum += NN_floatToHalf(
            NN_halfToFloat(((float16_t *)a->data)[i * a->shape[1] + k])
          * NN_halfToFloat(((float16_t *)b->data)[j * b->shape[1] + k])
        );
      }
      ((float16_t *)out->data)[i * out->shape[1] + j] = sum;
    }
  }
}

void NN_matmulT_F32(Tensor *out, Tensor *a, Tensor *b) {
  // currently only support 2D matrix multiplication
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(a->shape[1] == b->shape[1]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[0]);

  for (size_t i = 0; i < out->shape[0]; i += 1) {
    for (size_t j = 0; j < out->shape[1]; j += 1) {
      float sum = 0;
      for (size_t k = 0; k < a->shape[1]; k += 1) {
        sum += ((float *)a->data)[i * a->shape[1] + k] * ((float *)b->data)[j * b->shape[1] + k];
      }
      ((float *)out->data)[i * out->shape[1] + j] = sum;
    }
  }
}

void NN_matmul_I8_I8_I32(Tensor *out, Tensor *a, Tensor *b) {
  // currently only support 2D matrix multiplication
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_I8);
  assert(b->dtype == DTYPE_I8);
  assert(out->dtype == DTYPE_I8);
  assert(a->shape[1] == b->shape[0]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[1]);
  
  for (size_t i = 0; i < out->shape[0]; i += 1) {
    for (size_t j = 0; j < out->shape[1]; j += 1) {
      int32_t sum = 0;
      for (size_t k = 0; k < a->shape[1]; k += 1) {
        sum += ((int8_t *)a->data)[i * a->shape[1] + k] * ((int8_t *)b->data)[k * b->shape[1] + j];
      }
      ((int32_t *)out->data)[i * out->shape[1] + j] = sum;
    }
  }
}

void NN_matmul_I32(Tensor *out, Tensor *a, Tensor *b) {
  // currently only support 2D matrix multiplication
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(b->shape[0] == a->shape[1]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[1]);
  
  for (size_t i = 0; i < out->shape[0]; i += 1) {
    for (size_t j = 0; j < out->shape[1]; j += 1) {
      int32_t sum = 0;
      for (size_t k = 0; k < a->shape[1]; k += 1) {
        sum += ((int32_t *)a->data)[i * a->shape[1] + k] * ((int32_t *)b->data)[k * b->shape[1] + j];
      }
      ((int32_t *)out->data)[i * out->shape[1] + j] = sum;
    }
  }
}
