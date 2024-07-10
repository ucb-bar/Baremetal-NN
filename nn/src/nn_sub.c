
#include "nn_sub.h"


void NN_sub(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim == a->ndim);
  assert(out->ndim == a->ndim);

  switch (out->ndim) {
    case 1:
      NN_sub_1d(out, a, b);
      return;

    case 2:
      NN_sub_2d(out, a, b);
      return;
      
    case 3:
      NN_sub_3d(out, a, b);
      return;
      
    case 4:
      NN_sub_4d(out, a, b);
      return;
    
    default:
      printf("[ERROR] Unsupported tensor dimension: %lu\n", out->ndim);
      return;
  }
}

void NN_sub_1d(Tensor *out, Tensor *a, Tensor *b) {
  assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);

  if (out->dtype == DTYPE_F32 && a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32) {
    NN__sub_f32(out->size, (float *)out->data, (float *)a->data, (float *)b->data);
    return;
  }
  
  if (out->dtype == DTYPE_I8 && a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8) {
    NN__sub_i8(out->size, (int8_t *)out->data, (int8_t *)a->data, (int8_t *)b->data);
    return;
  }

  printf("[ERROR] Unsupported operation between tensor with dtype %s = %s - %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

void NN_sub_2d(Tensor *out, Tensor *a, Tensor *b) {
  assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);
  assert(out->shape[1] == a->shape[1] || out->shape[1] == b->shape[1]);

  if (a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]) {
    if (out->dtype == DTYPE_F32 && a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32) {
      NN__sub_f32(out->size, (float *)out->data, (float *)a->data, (float *)b->data);
      return;
    }
    if (out->dtype == DTYPE_I8 && a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8) {
      NN__sub_i8(out->size, (int8_t *)out->data, (int8_t *)a->data, (int8_t *)b->data);
      return;
    }
    if (out->dtype == DTYPE_I16 && a->dtype == DTYPE_I16 && b->dtype == DTYPE_I16) {
      NN__sub_i16(out->size, (int16_t *)out->data, (int16_t *)a->data, (int16_t *)b->data);
      return;
    }
    if (out->dtype == DTYPE_I32 && a->dtype == DTYPE_I32 && b->dtype == DTYPE_I32) {
      NN__sub_i32(out->size, (int32_t *)out->data, (int32_t *)a->data, (int32_t *)b->data);
      return;
    }
    if (out->dtype == DTYPE_U8 && a->dtype == DTYPE_U8 && b->dtype == DTYPE_U8) {
      NN__sub_u8(out->size, (uint8_t *)out->data, (uint8_t *)a->data, (uint8_t *)b->data);
      return;
    }
  }

  if (out->dtype == DTYPE_F32 && a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32) {
    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        // handle broadcasting
        size_t a_i = i < a->shape[0] ? i : 0;
        size_t a_j = j < a->shape[1] ? j : 0;
        size_t b_i = i < b->shape[0] ? i : 0;
        size_t b_j = j < b->shape[1] ? j : 0;

        ((float *)out->data)[i * out->shape[1] + j]
          = ((float *)a->data)[a_i * a->shape[1] + a_j]
          - ((float *)b->data)[b_i * b->shape[1] + b_j];
      }
    }
    return;
  }

  printf("[ERROR] Unsupported operation between tensor with dtype %s = %s - %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

void NN_sub_3d(Tensor *out, Tensor *a, Tensor *b) {
  assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);
  assert(out->shape[1] == a->shape[1] || out->shape[1] == b->shape[1]);
  assert(out->shape[2] == a->shape[2] || out->shape[2] == b->shape[2]);

  if (out->dtype == DTYPE_F32 && a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32) {
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

          ((float *)out->data)[i * out->shape[1] * out->shape[2] + j * out->shape[2] + k]
            = ((float *)a->data)[a_i * a->shape[1] * a->shape[2] + a_j * a->shape[2] + a_k]
            - ((float *)b->data)[b_i * b->shape[1] * b->shape[2] + b_j * b->shape[2] + b_k];
        }
      }
    }
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = %s - %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

void NN_sub_4d(Tensor *out, Tensor *a, Tensor *b) {
  assert(out->shape[0] == a->shape[0] || out->shape[0] == b->shape[0]);
  assert(out->shape[1] == a->shape[1] || out->shape[1] == b->shape[1]);
  assert(out->shape[2] == a->shape[2] || out->shape[2] == b->shape[2]);
  assert(out->shape[3] == a->shape[3] || out->shape[3] == b->shape[3]);

  if (out->dtype == DTYPE_F32 && a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32) {
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

            ((float *)out->data)[n * out->shape[1] * out->shape[2] * out->shape[3] + c * out->shape[2] * out->shape[3] + h * out->shape[3] + w]
              = ((float *)a->data)[a_n * a->shape[1] * a->shape[2] * a->shape[3] + a_c * a->shape[2] * a->shape[3] + a_h * a->shape[3] + a_w]
              - ((float *)b->data)[b_n * b->shape[1] * b->shape[2] * b->shape[3] + b_c * b->shape[2] * b->shape[3] + b_h * b->shape[3] + b_w];
          }
        }
      }
    }
  }

  printf("[ERROR] Unsupported operation between tensor with dtype %s = %s - %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

