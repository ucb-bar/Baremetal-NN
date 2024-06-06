
#include "nn_add.h"


void NN_add(Tensor *out, Tensor *a, Tensor *b) {
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32) {
    NN_add_F32(out, a, b);
    return;
  }
  if ((a->dtype == DTYPE_I8 || a->dtype == DTYPE_I32) && (b->dtype == DTYPE_I8 || b->dtype == DTYPE_I32)) {
    NN_add_INT(out, a, b);
    return;
  }

  printf("Unsupported operation: %s + %s -> %s\n", NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype), NN_getDataTypeName(out->dtype));
}

void NN_add_F32(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->ndim == b->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(a->shape[0] == b->shape[0]);
  
  out->dtype = DTYPE_F32;

  uint8_t *a_ptr = a->data;
  uint8_t *b_ptr = b->data;
  uint8_t *out_ptr = out->data;

  for (size_t i = 0; i<out->ndim; i+=1) {
    out->shape[i] = a->shape[i] > b->shape[i] ? a->shape[i] : b->shape[i];
  }

  switch (out->ndim) {
    case 1:
      for (size_t i = 0; i<out->shape[0]; i+=1) {
        *((float *)out_ptr) = *((float *)a_ptr) + *((float *)b_ptr);
        a_ptr += a->strides[0];
        b_ptr += b->strides[0];
        out_ptr += out->strides[0];
      }
      return;
    case 2:
      for (size_t i = 0; i<out->shape[0]; i+=1) {
        for (size_t j = 0; j<out->shape[1]; j+=1) {
          *((float *)out_ptr) = *((float *)a_ptr) + *((float *)b_ptr);
          a_ptr += a->strides[1];
          b_ptr += b->strides[1];
          out_ptr += out->strides[1];
        }
        a_ptr -= a->strides[1] * a->shape[1];
        b_ptr -= b->strides[1] * b->shape[1];
        a_ptr += a->strides[0];
        b_ptr += b->strides[0];
      }
      return;
  }
  
  printf("Unsupported operation between tensor with shape ");
  NN_printShape(a->shape);
  printf(" and ");
  NN_printShape(b->shape);
  printf("\n");
}

void NN_add_INT(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->ndim == b->ndim);
  assert(a->dtype == DTYPE_I8 || a->dtype == DTYPE_I32);
  assert(b->dtype == DTYPE_I8 || b->dtype == DTYPE_I32);
  assert(a->shape[0] == b->shape[0]);

  out->dtype = DTYPE_I32;
  
  uint8_t *a_ptr = a->data;
  uint8_t *b_ptr = b->data;
  uint8_t *out_ptr = out->data;

  for (size_t i = 0; i<out->ndim; i+=1) {
    out->shape[i] = a->shape[i] > b->shape[i] ? a->shape[i] : b->shape[i];
  }
  
  switch (out->ndim) {
    case 1:
      for (size_t i = 0; i<out->shape[0]; i+=1) {
        *((int32_t *)out_ptr) = *((int32_t *)a_ptr) + *((int32_t *)b_ptr);
        a_ptr += a->strides[0];
        b_ptr += b->strides[0];
        out_ptr += out->strides[0];
      }
      return;
    case 2:
      for (size_t i = 0; i<out->shape[0]; i+=1) {
        for (size_t j = 0; j<out->shape[1]; j+=1) {
          *((int32_t *)out_ptr) = *((int32_t *)a_ptr) + *((int32_t *)b_ptr);
          a_ptr += a->strides[1];
          b_ptr += b->strides[1];
          out_ptr += out->strides[1];
        }
        a_ptr -= a->strides[1] * a->shape[1];
        b_ptr -= b->strides[1] * b->shape[1];
        a_ptr += a->strides[0];
        b_ptr += b->strides[0];
      }
      return;
  }

  printf("Unsupported operation between tensor with shape ");
  NN_printShape(a->shape);
  printf(" and ");
  NN_printShape(b->shape);
  printf("\n");
}

