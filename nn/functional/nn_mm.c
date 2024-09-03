
#include "nn_mm.h"


void NN_mm(Tensor *out, const Tensor *a, const Tensor *b) {
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->shape[1] == b->shape[0]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[1]);

  #ifdef GEMMINI
    NN_mm_f32(out->shape[0], out->shape[1], (float *)out->data, (float *)a->data, (float *)b->data);
    return;
  #endif

  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        NN_dot_f32(a->shape[1], 
          (float *)out->data + i * out->shape[1] + j, 
          (float *)a->data + i * a->shape[1], 1,
          (float *)b->data + j, b->shape[1]
          );
      }
    }
    return;
  }
  if (a->dtype == DTYPE_F16 && b->dtype == DTYPE_F16 && out->dtype == DTYPE_F16) {
    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        NN_dot_f16(a->shape[1], 
          (float16_t *)out->data + i * out->shape[1] + j, 
          (float16_t *)a->data + i * a->shape[1], 1,
          (float16_t *)b->data + j, b->shape[1]
          );
      }
    }
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

void NN_mm_t(Tensor *out, const Tensor *a, const Tensor *b) {
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->shape[1] == b->shape[1]);
  assert(out->shape[0] == a->shape[0]);
  assert(out->shape[1] == b->shape[0]);

  if (a->dtype == DTYPE_F16 && b->dtype == DTYPE_F16 && out->dtype == DTYPE_F16) {
    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        NN_dot_f16(a->shape[1],
          (float16_t *)out->data + i * out->shape[1] + j, 
          (float16_t *)a->data + i * a->shape[1], 1,
          (float16_t *)b->data + j * b->shape[1], 1
          );
      }
    }
    return;
  }
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        NN_dot_f32(a->shape[1], 
          (float *)out->data + i * out->shape[1] + j,
          (float *)a->data + i * a->shape[1], 1,
          (float *)b->data + j * b->shape[1], 1
          );
      }
    }
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

