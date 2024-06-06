
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
  assert(a->shape[1] == b->shape[0]);
  
  out->dtype = DTYPE_F32;
  out->shape[0] = a->shape[0];
  out->shape[1] = b->shape[1];

  uint8_t *out_ptr = out->data;
  uint8_t *a_ptr = a->data;
  uint8_t *b_ptr = b->data;

  for (size_t i = 0; i < a->shape[0]; i++) {
    for (size_t j = 0; j < b->shape[1]; j++) {
      *((float *)out_ptr) = 0; // Initialize the output element
      for (size_t k = 0; k < a->shape[1]; k++) {
        *((float *)out_ptr) += *((float *)(a_ptr)) * *((float *)(b_ptr));
        a_ptr += a->strides[1];
        b_ptr += b->strides[0];
      }
      out_ptr += out->strides[1];           // Move to the next column in the output matrix
      a_ptr -= a->strides[1] * a->shape[1]; // Move back to the first element in the row of the first matrix
      b_ptr -= b->strides[0] * b->shape[0]; // Move back to the first element in the column of the second matrix
      b_ptr += b->strides[1];              // Move to the next element in the column of the second matrix
    }
    out_ptr -= out->strides[1] * b->shape[1];
    out_ptr += out->strides[0]; // Move to the next row in the output matrix
    a_ptr += a->strides[0];
    b_ptr -= b->strides[1] * b->shape[1]; // Move back to the first element in the column of the second matrix
  }

  // // simple sanity check version
  // for (size_t i = 0; i < a->shape[0]; i+=1) {
  //   for (size_t j = 0; j < b->shape[1]; j+=1) {
  //     float sum = 0;
  //     for (size_t k = 0; k < a->shape[1]; k+=1) {
  //       sum += ((float *)a->data)[i * a->shape[1] + k] * ((float *)b->data)[k * b->shape[1] + j];
  //     }
  //     ((float *)out->data)[i * out->shape[1] + j] = sum;
  //   }
  // }   
}

void NN_matmul_I8_I8_I32(Tensor *out, Tensor *a, Tensor *b) {
  // currently only support 2D matrix multiplication
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_I8);
  assert(b->dtype == DTYPE_I8);
  assert(a->shape[1] == b->shape[0]);
  
  out->dtype = DTYPE_I32;

  out->shape[0] = a->shape[0];
  out->shape[1] = b->shape[1];

  int8_t *out_ptr = out->data;
  int8_t *a_ptr = a->data;
  int8_t *b_ptr = b->data;

  for (size_t i = 0; i < a->shape[0]; i++) {
    for (size_t j = 0; j < b->shape[1]; j++) {
      *((int32_t *)out_ptr) = 0; // Initialize the output element
      for (size_t k = 0; k < a->shape[1]; k++) {
        *((int32_t *)out_ptr) += *((int8_t *)(a_ptr)) * *((int8_t *)(b_ptr));
        a_ptr += a->strides[1];
        b_ptr += b->strides[0];
      }
      a_ptr -= a->strides[1] * a->shape[1]; // Move back to the first element in the row of the first matrix
      b_ptr -= b->strides[0] * b->shape[0]; // Move back to the first element in the column of the second matrix
      b_ptr += b->strides[1];              // Move to the next element in the column of the second matrix
      out_ptr += out->strides[1];           // Move to the next column in the output matrix
    }
    b_ptr -= b->strides[1] * b->shape[1]; // Move back to the first element in the column of the second matrix
    out_ptr -= out->strides[1] * b->shape[1];
    a_ptr += a->strides[0];
    out_ptr += out->strides[0]; // Move to the next row in the output matrix
  }
}

void NN_matmul_I32(Tensor *out, Tensor *a, Tensor *b) {
  // currently only support 2D matrix multiplication
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->dtype == DTYPE_I32);
  assert(b->dtype == DTYPE_I32);
  assert(a->shape[1] == b->shape[0]);
  
  out->dtype = DTYPE_I32;

  out->shape[0] = a->shape[0];
  out->shape[1] = b->shape[1];

  int8_t *out_ptr = out->data;
  int8_t *a_ptr = a->data;
  int8_t *b_ptr = b->data;

  for (size_t i = 0; i < a->shape[0]; i++) {
    for (size_t j = 0; j < b->shape[1]; j++) {
      *((int32_t *)out_ptr) = 0; // Initialize the output element
      for (size_t k = 0; k < a->shape[1]; k++) {
        *((int32_t *)out_ptr) += *((int32_t *)(a_ptr)) * *((int32_t *)(b_ptr));
        a_ptr += a->strides[1];
        b_ptr += b->strides[0];
      }
      a_ptr -= a->strides[1] * a->shape[1]; // Move back to the first element in the row of the first matrix
      b_ptr -= b->strides[0] * b->shape[0]; // Move back to the first element in the column of the second matrix
      b_ptr += b->strides[1];              // Move to the next element in the column of the second matrix
      out_ptr += out->strides[1];           // Move to the next column in the output matrix
    }
    b_ptr -= b->strides[1] * b->shape[1]; // Move back to the first element in the column of the second matrix
    out_ptr -= out->strides[1] * b->shape[1];
    a_ptr += a->strides[0];
    out_ptr += out->strides[0]; // Move to the next row in the output matrix
  }
}
