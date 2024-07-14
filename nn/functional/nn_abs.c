
#include "nn_abs.h"


void NN_abs(Tensor *out, const Tensor *in) {
  assert(out->ndim == in->ndim);
  assert(out->dtype == in->dtype);
  assert(out->size == in->size);
  
  switch (out->dtype) {
    case DTYPE_I8:
      NN__abs_i8(out->size, (int8_t *)out->data, 1, (int8_t *)in->data, 1);
      return;
    case DTYPE_I16:
      NN__abs_i16(out->size, (int16_t *)out->data, 1, (int16_t *)in->data, 1);
      return;
    case DTYPE_I32:
      NN__abs_i32(out->size, (int32_t *)out->data, 1, (int32_t *)in->data, 1);
      return;
    case DTYPE_F16:
      NN__abs_f16(out->size, (float16_t *)out->data, 1, (float16_t *)in->data, 1);
      return;
    case DTYPE_F32:
      NN__abs_f32(out->size, (float *)out->data, 1, (float *)in->data, 1);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation of tensor with dtype %s = |%s|\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(in->dtype)
  );
}

void NN_abs_inplace(Tensor *x) {
  NN_abs(x, x);
}
