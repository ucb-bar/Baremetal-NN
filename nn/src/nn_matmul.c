
#include "nn_matmul.h"

void NN_matmul(Tensor *out, Tensor *a, Tensor *b) {
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
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
    return;
  }
  if (a->dtype == DTYPE_F16 && b->dtype == DTYPE_F16 && out->dtype == DTYPE_F16) {
    // currently only support 2D matrix multiplication
    assert(a->ndim == 2);
    assert(b->ndim == 2);
    assert(a->dtype == DTYPE_F16);
    assert(b->dtype == DTYPE_F16);
    assert(out->dtype == DTYPE_F16);
    assert(a->shape[1] == b->shape[0]);
    assert(out->shape[0] == a->shape[0]);
    assert(out->shape[1] == b->shape[1]);

    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        float sum = 0;
        for (size_t k = 0; k < a->shape[1]; k += 1) {
          sum += NN_halfToFloat(((float16_t *)a->data)[i * a->shape[1] + k]) * NN_halfToFloat(((float16_t *)b->data)[k * b->shape[1] + j]);
        }
        ((float16_t *)out->data)[i * out->shape[1] + j] = NN_floatToHalf(sum);
      }
    }
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype)
  );
}

void NN_matmulT(Tensor *out, Tensor *a, Tensor *b) {
  if (a->dtype == DTYPE_F16 && b->dtype == DTYPE_F16 && out->dtype == DTYPE_F16) {
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
        NN__dot_F16(a->shape[1], 
          (float16_t *)out->data + i * out->shape[1] + j, 
          (float16_t *)a->data + i * a->shape[1], 
          (float16_t *)b->data + j * b->shape[1]
          );
      }
    }
    return;
  }
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
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
        NN__dot_F32(a->shape[1], 
          (float *)out->data + i * out->shape[1] + j, 
          (float *)a->data + i * a->shape[1], 
          (float *)b->data + j * b->shape[1]
          );
      }
    }
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype)
  );
}

