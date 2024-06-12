
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

  switch (out->ndim) {
    case 1:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);

      for (size_t i = 0; i < out->shape[0]; i += 1) {
        ((float *)out->data)[i] = ((float *)a->data)[i] + ((float *)b->data)[i];
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

          ((float *)out->data)[i * out->shape[1] + j] = ((float *)a->data)[a_i * a->shape[1] + a_j] + ((float *)b->data)[b_i * b->shape[1] + b_j];
        }
      }
      return;
    
    case 3:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);
      assert(out->shape[1] == a->shape[1] || out->shape[1] == b->shape[1]);
      assert(out->shape[2] == a->shape[2] || out->shape[2] == b->shape[2]);

      for (size_t i = 0; i < out->shape[0]; i += 1) {
        for (size_t j = 0; j < out->shape[1]; j += 1) {
          for (size_t k = 0; k < out->shape[2]; k += 1) {
            // handle broadcasting
            size_t a_i = i < a->shape[0] ? i : 0;
            size_t a_j = j < a->shape[1] ? j : 0;
            size_t a_k = k < a->shape[2] ? k : 0;
            size_t b_i = i < b->shape[0] ? i : 0;
            size_t b_j = j < b->shape[1] ? j : 0;
            size_t b_k = k < b->shape[2] ? k : 0;

            ((float *)out->data)[i * out->shape[1] * out->shape[2] + j * out->shape[2] + k] = ((float *)a->data)[a_i * a->shape[1] * a->shape[2] + a_j * a->shape[2] + a_k] + ((float *)b->data)[b_i * b->shape[1] * b->shape[2] + b_j * b->shape[2] + b_k];
          }
        }
      }
      return;
    
    case 4:
      assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);
      assert(out->shape[1] == a->shape[1] || out->shape[1] == b->shape[1]);
      assert(out->shape[2] == a->shape[2] || out->shape[2] == b->shape[2]);
      assert(out->shape[3] == a->shape[3] || out->shape[3] == b->shape[3]);

      for (size_t n = 0; n < out->shape[0]; n += 1) {
        for (size_t c = 0; c < out->shape[1]; c += 1) {
          for (size_t h = 0; h < out->shape[2]; h += 1) {
            for (size_t w = 0; w < out->shape[3]; w += 1) {
              // handle broadcasting
              size_t a_n = n < a->shape[0] ? n : 0;
              size_t a_c = c < a->shape[1] ? c : 0;
              size_t a_h = h < a->shape[2] ? h : 0;
              size_t a_w = w < a->shape[3] ? w : 0;
              size_t b_n = n < b->shape[0] ? n : 0;
              size_t b_c = c < b->shape[1] ? c : 0;
              size_t b_h = h < b->shape[2] ? h : 0;
              size_t b_w = w < b->shape[3] ? w : 0;

              ((float *)out->data)[n * out->shape[1] * out->shape[2] * out->shape[3] + c * out->shape[2] * out->shape[3] + h * out->shape[3] + w] = ((float *)a->data)[a_n * a->shape[1] * a->shape[2] * a->shape[3] + a_c * a->shape[2] * a->shape[3] + a_h * a->shape[3] + a_w] + ((float *)b->data)[b_n * b->shape[1] * b->shape[2] * b->shape[3] + b_c * b->shape[2] * b->shape[3] + b_h * b->shape[3] + b_w];
            }
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

void NN_add_INT(Tensor *out, Tensor *a, Tensor *b) {
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
          ((int8_t *)out->data)[i] = ((int8_t *)a->data)[i] + ((int8_t *)b->data)[i];
        } else if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I32) {
          ((int32_t *)out->data)[i] = ((int32_t *)a->data)[i] + ((int32_t *)b->data)[i];
        } else if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I32) {
          ((int32_t *)out->data)[i] = (uint32_t)(((int8_t *)a->data))[i] + ((int32_t *)b->data)[i];
        } else if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I8) {
          ((int32_t *)out->data)[i] = ((int32_t *)a->data)[i] + (uint32_t)(((int8_t *)b->data)[i]);
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
            ((int8_t *)out->data)[i * out->shape[1] + j] = ((int8_t *)a->data)[a_i * a->shape[1] + a_j] + ((int8_t *)b->data)[b_i * b->shape[1] + b_j];
          } else if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I32) {
            ((int32_t *)out->data)[i * out->shape[1] + j] = ((int32_t *)a->data)[a_i * a->shape[1] + a_j] + ((int32_t *)b->data)[b_i * b->shape[1] + b_j];
          } else if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I32) {
            ((int32_t *)out->data)[i * out->shape[1] + j] = (uint32_t)(((int8_t *)a->data))[a_i * a->shape[1] + a_j] + ((int32_t *)b->data)[b_i * b->shape[1] + b_j];
          } else if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I8) {
            ((int32_t *)out->data)[i * out->shape[1] + j] = ((int32_t *)a->data)[a_i * a->shape[1] + a_j] + (uint32_t)(((int8_t *)b->data)[b_i * b->shape[1] + b_j]);
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

void NN_addF_F32(Tensor *out, Tensor *in, float scalar) {
  assert(out->ndim == in->ndim);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == in->size);
  
  for (size_t i = 0; i < out->size; i += 1) {
    ((float *)out->data)[i] = ((float *)in->data)[i] + scalar;
  }
}

