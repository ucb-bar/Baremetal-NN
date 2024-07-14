
#include "nn_mv.h"


void NN_mv(Tensor *out, const Tensor *a, const Tensor *v) {
  assert(a->ndim == 2);
  assert(v->ndim == 1);
  assert(v->shape[0] == a->shape[1]);
  assert(out->shape[0] == a->shape[0]);

  if (a->dtype == DTYPE_F16 && v->dtype == DTYPE_F16 && out->dtype == DTYPE_F16) {
    for (size_t i = 0; i < out->shape[0]; i += 1) {
      NN__dot_f16(a->shape[1], 
        (float16_t *)out->data + i, 
        (float16_t *)a->data + i * a->shape[1], 1,
        (float16_t *)v->data, 1
        );
    }
    return;
  }
  if (a->dtype == DTYPE_F32 && v->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
    for (size_t i = 0; i < out->shape[0]; i += 1) {
      NN__dot_f32(a->shape[1],
        (float *)out->data + i, 
        (float *)a->data + i * a->shape[1], 1,
        (float *)v->data, 1
        );
    }
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(v->dtype)
  );
}
