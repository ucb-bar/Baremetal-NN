
#include "fill.h"


void NN_fill(Tensor *tensor, float value) {
  switch (tensor->dtype) {
    case DTYPE_U8:
      NN_fill_u8(tensor->size, (uint8_t *)tensor->data, 1, (uint8_t)value);
      return;
    case DTYPE_I8:
      NN_fill_i8(tensor->size, (int8_t *)tensor->data, 1, (int8_t)value);
      return;
    case DTYPE_I32:
      NN_fill_i32(tensor->size, (int32_t *)tensor->data, 1, (int32_t)value);
      return;
    case DTYPE_F16:
      NN_fill_f16(tensor->size, (float16_t *)tensor->data, 1, NN_float_to_half(value));
      return;
    case DTYPE_F32:
      NN_fill_f32(tensor->size, (float *)tensor->data, 1, value);
      return;
    default:
      printf("[ERROR] Unsupported operation fill to tensor with dtype: %d\n", tensor->dtype);
  }
}
