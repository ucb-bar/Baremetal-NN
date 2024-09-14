
#include "sum.h"


void NN_sum(Tensor *out, const Tensor *tensor) {
  int32_t result_i32;
  switch (tensor->dtype) {
    case DTYPE_U8:
      NN_sum_u8_to_i32(tensor->size, &result_i32, (uint8_t *)tensor->data, 1);
      switch (out->dtype) {
        case DTYPE_U16:
          *(uint16_t *)out->data = (uint16_t)result_i32;
          return;
        case DTYPE_U32:
          *(uint32_t *)out->data = (uint32_t)result_i32;
          return;
        case DTYPE_I32:
          *(int32_t *)out->data = result_i32;
          return;
        default:
          break;
      }
      break;

    case DTYPE_I16:
      NN_sum_i16_to_i32(tensor->size, &result_i32, (int16_t *)tensor->data, 1);
      switch (out->dtype) {
        case DTYPE_I16:
          *(int16_t *)out->data = (int16_t)result_i32;
          return;
        case DTYPE_I32:
          *(int32_t *)out->data = result_i32;
          return;
        default:
          break;
      }
      break;

    case DTYPE_I32:
      switch (out->dtype) {
        case DTYPE_I32:
          NN_sum_i32(tensor->size, (int32_t *)out->data, (int32_t *)tensor->data, 1);
          return;
        default:
          break;
      }
      break;
    
    case DTYPE_F32:
      switch (out->dtype) {
        case DTYPE_F32:
          NN_sum_f32(tensor->size, (float *)out->data, (float *)tensor->data, 1);
          return;
        default:
          break;
      }
    
    default:
      break;
  }

  printf("[ERROR] Unsupported operation of tensor with dtype %s = sum(%s)\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(tensor->dtype)
  );
}
