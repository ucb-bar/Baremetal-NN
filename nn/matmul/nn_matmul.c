
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
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(a->shape[1] == b->shape[0]);
  
  out->dtype = DTYPE_F32;
  out->shape[0] = a->shape[0];
  out->shape[1] = b->shape[1];

  float *out_data = (float *)out->data;
  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;

  for (size_t i=0; i<out->shape[0]; i+=1) {
    for (size_t j=0; j<out->shape[1]; j+=1) {
      out_data[i*out->shape[1]+j] = 0;
      for (size_t k=0; k<a->shape[1]; k+=1) {
        out_data[i*out->shape[1]+j] += a_data[i*a->shape[1]+k] * b_data[k*b->shape[1]+j];
      }
    }
  }
}

void NN_matmul_I8_I8_I32(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->dtype == DTYPE_I8);
  assert(b->dtype == DTYPE_I8);
  assert(a->shape[1] == b->shape[0]);
  
  out->dtype = DTYPE_I32;
  out->shape[0] = a->shape[0];
  out->shape[1] = b->shape[1];

  int32_t *out_data = (int32_t *)out->data;
  int8_t *a_data = (int8_t *)a->data;
  int8_t *b_data = (int8_t *)b->data;

  for (size_t i=0; i<out->shape[0]; i+=1) {
    for (size_t j=0; j<out->shape[1]; j+=1) {
      out_data[i*out->shape[1]+j] = 0;
      for (size_t k=0; k<a->shape[1]; k+=1) {
        out_data[i*out->shape[1]+j] += a_data[i*a->shape[1]+k] * b_data[k*b->shape[1]+j];
      }
    }
  }
}

void NN_matmul_I32(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->dtype == DTYPE_I32);
  assert(b->dtype == DTYPE_I32);
  assert(a->shape[1] == b->shape[0]);
  
  out->dtype = DTYPE_I32;
  out->shape[0] = a->shape[0];
  out->shape[1] = b->shape[1];

  int32_t *out_data = (int32_t *)out->data;
  int32_t *a_data = (int32_t *)a->data;
  int32_t *b_data = (int32_t *)b->data;

  for (size_t i=0; i<out->shape[0]; i+=1) {
    for (size_t j=0; j<out->shape[1]; j+=1) {
      out_data[i*out->shape[1]+j] = 0;
      for (size_t k=0; k<a->shape[1]; k+=1) {
        out_data[i*out->shape[1]+j] += a_data[i*a->shape[1]+k] * b_data[k*b->shape[1]+j];
      }
    }
  }
}
