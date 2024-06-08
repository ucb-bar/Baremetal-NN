
#include "nn_sub.h"


void NN_sub(Tensor *out, Tensor *a, Tensor *b) {
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32) {
    NN_sub_F32(out, a, b);
    return;
  }
  if ((a->dtype == DTYPE_I8 || a->dtype == DTYPE_I32) && (b->dtype == DTYPE_I8 || b->dtype == DTYPE_I32)) {
    NN_sub_INT(out, a, b);
    return;
  }

  printf("Unsupported operation: %s + %s -> %s\n", NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype), NN_getDataTypeName(out->dtype));
}

void NN_sub_F32(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  
  switch (out->ndim) {
    case 1:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);

      for (size_t i = 0; i < out->shape[0]; i += 1) {
        ((float *)out->data)[i] = ((float *)a->data)[i] - ((float *)b->data)[i];
      }
      return;
    case 2:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);
      assert(out->shape[1] == a->shape[1] || out->shape[1] == b->shape[1]);

      for (size_t i = 0; i < out->shape[0]; i += 1) {
        for (size_t j = 0; j < out->shape[1]; j += 1) {
          // handle broadcasting
          size_t a_i = i < a->shape[0] ? i : 0;
          size_t a_j = j < a->shape[1] ? j : 0;
          size_t b_i = i < b->shape[0] ? i : 0;
          size_t b_j = j < b->shape[1] ? j : 0;

          ((float *)out->data)[i * out->shape[1] + j] = ((float *)a->data)[a_i * a->shape[1] + a_j] - ((float *)b->data)[b_i * b->shape[1] + b_j];
        }
      }
      return;
  }
  
  printf("Unsupported operation between tensor with shape ");
  NN_printShape(a);
  printf(" and ");
  NN_printShape(b);
  printf("\n");
}

void NN_sub_INT(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);
  assert(a->dtype == DTYPE_I8 || a->dtype == DTYPE_I32);
  assert(b->dtype == DTYPE_I8 || b->dtype == DTYPE_I32);
  assert((out->dtype == DTYPE_I32) || (out->dtype == DTYPE_I8 && (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8)));
  
  switch (out->ndim) {
    case 1:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);

      for (size_t i = 0; i < out->shape[0]; i += 1) {
        if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8) {
          ((int8_t *)out->data)[i] = ((int8_t *)a->data)[i] - ((int8_t *)b->data)[i];
        } else if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I32) {
          ((int32_t *)out->data)[i] = ((int32_t *)a->data)[i] - ((int32_t *)b->data)[i];
        } else if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I32) {
          ((int32_t *)out->data)[i] = (uint32_t)(((int8_t *)a->data))[i] - ((int32_t *)b->data)[i];
        } else if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I8) {
          ((int32_t *)out->data)[i] = ((int32_t *)a->data)[i] - (uint32_t)(((int8_t *)b->data)[i]);
        }
      }
      return;
    
    case 2:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);
      assert(out->shape[1] == a->shape[1] || out->shape[1] == b->shape[1]);

      for (size_t i = 0; i < out->shape[0]; i += 1) {
        for (size_t j = 0; j < out->shape[1]; j += 1) {
          // handle broadcasting
          size_t a_i = i < a->shape[0] ? i : 0;
          size_t a_j = j < a->shape[1] ? j : 0;
          size_t b_i = i < b->shape[0] ? i : 0;
          size_t b_j = j < b->shape[1] ? j : 0;

          if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8) {
            ((int8_t *)out->data)[i * out->shape[1] + j] = ((int8_t *)a->data)[a_i * a->shape[1] + a_j] - ((int8_t *)b->data)[b_i * b->shape[1] + b_j];
          } else if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I32) {
            ((int32_t *)out->data)[i * out->shape[1] + j] = ((int32_t *)a->data)[a_i * a->shape[1] + a_j] - ((int32_t *)b->data)[b_i * b->shape[1] + b_j];
          } else if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I32) {
            ((int32_t *)out->data)[i * out->shape[1] + j] = (uint32_t)(((int8_t *)a->data))[a_i * a->shape[1] + a_j] - ((int32_t *)b->data)[b_i * b->shape[1] + b_j];
          } else if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I8) {
            ((int32_t *)out->data)[i * out->shape[1] + j] = ((int32_t *)a->data)[a_i * a->shape[1] + a_j] - (uint32_t)(((int8_t *)b->data)[b_i * b->shape[1] + b_j]);
          }
        }
      }
      return;
  }

  printf("Unsupported operation between tensor with shape ");
  NN_printShape(a);
  printf(" and ");
  NN_printShape(b);
  printf("\n");
}

