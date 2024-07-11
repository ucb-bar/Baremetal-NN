
#include "nn_mv.h"


void NN_mv(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->ndim == 2);
  assert(b->ndim == 1);
  assert(b->shape[0] == a->shape[1]);
  assert(out->shape[0] == a->shape[0]);

  if (a->dtype == DTYPE_F16 && b->dtype == DTYPE_F16 && out->dtype == DTYPE_F16) {
    for (size_t i = 0; i < out->shape[0]; i += 1) {
      NN__dot_f16(a->shape[1], 
        (float16_t *)out->data + i, 
        (float16_t *)a->data + i * a->shape[1], 1,
        (float16_t *)b->data, 1
        );
    }
    return;
  }
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
    for (size_t i = 0; i < out->shape[0]; i += 1) {
      NN__dot_f32(a->shape[1],
        (float *)out->data + i, 
        (float *)a->data + i * a->shape[1], 1,
        (float *)b->data, 1
        );
    }
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}
