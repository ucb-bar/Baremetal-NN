
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
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  
  uint8_t *out_ptr = out->data;
  uint8_t *a_ptr = a->data;
  uint8_t *b_ptr = b->data;

  switch (out->ndim) {
    case 1:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);

      for (size_t i = 0; i < out->shape[0]; i += 1) {
        *((float *)out_ptr) = *((float *)a_ptr) + *((float *)b_ptr);
        out_ptr += out->strides[0];
        a_ptr += a->strides[0];
        b_ptr += b->strides[0];
      }
      return;
    case 2:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);
      assert(out->shape[1] == a->shape[1] || out->shape[1] == b->shape[1]);

      for (size_t i = 0; i < out->shape[0]; i += 1) {
        for (size_t j = 0; j < out->shape[1]; j += 1) {
          *((float *)out_ptr) = *((float *)a_ptr) + *((float *)b_ptr);
          out_ptr += out->strides[1];
          a_ptr += a->strides[1];
          b_ptr += b->strides[1];
        }
        a_ptr -= a->strides[1] * a->shape[1];
        a_ptr += a->strides[0];
        b_ptr -= b->strides[1] * b->shape[1];
        b_ptr += b->strides[0];
      }
      return;
  }
  
  printf("Unsupported operation between tensor with shape ");
  NN_printShape(a);
  printf(" and ");
  NN_printShape(b);
  printf("\n");
}

void NN_add_INT(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_I8 || a->dtype == DTYPE_I32);
  assert(b->dtype == DTYPE_I8 || b->dtype == DTYPE_I32);
  
  uint8_t *out_ptr = out->data;
  uint8_t *a_ptr = a->data;
  uint8_t *b_ptr = b->data;

  switch (out->ndim) {
    case 1:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);

      for (size_t i = 0; i < out->shape[0]; i += 1) {
        *((int32_t *)out_ptr) = *((int32_t *)a_ptr) + *((int32_t *)b_ptr);
        out_ptr += out->strides[0];
        a_ptr += a->strides[0];
        b_ptr += b->strides[0];
      }
      return;
    case 2:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);
      assert(out->shape[1] == a->shape[1] || out->shape[1] == b->shape[1]);
      
      for (size_t i = 0; i < out->shape[0]; i += 1) {
        for (size_t j = 0; j < out->shape[1]; j += 1) {
          *((int32_t *)out_ptr) = *((int32_t *)a_ptr) + *((int32_t *)b_ptr);
          out_ptr += out->strides[1];
          a_ptr += a->strides[1];
          b_ptr += b->strides[1];
        }
        a_ptr -= a->strides[1] * a->shape[1];
        a_ptr += a->strides[0];
        b_ptr -= b->strides[1] * b->shape[1];
        b_ptr += b->strides[0];
      }
      return;
  }

  printf("Unsupported operation between tensor with shape ");
  NN_printShape(a);
  printf(" and ");
  NN_printShape(b);
  printf("\n");
}

