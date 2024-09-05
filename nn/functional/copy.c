
#include "copy.h"

void NN_copy(Tensor *dst, const Tensor *src) {
  assert(dst->ndim == src->ndim);
  assert(dst->size == src->size);
  
  if (dst->dtype == src->dtype) {
    memcpy(dst->data, src->data, dst->size * NN_sizeof(dst->dtype));
    return;
  }

  switch (src->dtype) {
    case DTYPE_U8:
      switch (dst->dtype) {
        case DTYPE_U16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint16_t *)dst->data)[i] = (uint16_t)((uint8_t *)src->data)[i];
          }
          return;
        case DTYPE_I16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int16_t *)dst->data)[i] = (int16_t)((uint8_t *)src->data)[i];
          }
          return;
        case DTYPE_U32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint32_t *)dst->data)[i] = (uint32_t)((uint8_t *)src->data)[i];
          }
          return;
        case DTYPE_I32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int32_t *)dst->data)[i] = (int32_t)((uint8_t *)src->data)[i];
          }
          return;
        case DTYPE_F16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((float16_t *)dst->data)[i] = NN_float_to_half((float)((uint8_t *)src->data)[i]);
          }
          return;
        case DTYPE_F32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((float *)dst->data)[i] = (float)((uint8_t *)src->data)[i];
          }
          return;
        default:
          break;
      }
      break;
    
    case DTYPE_I8:
      switch (dst->dtype) {
        case DTYPE_I16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int16_t *)dst->data)[i] = (int16_t)((int8_t *)src->data)[i];
          }
          return;
        case DTYPE_I32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int32_t *)dst->data)[i] = (int32_t)((int8_t *)src->data)[i];
          }
          return;
        case DTYPE_F32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((float *)dst->data)[i] = (float)((int8_t *)src->data)[i];
          }
          return;
        default:
          break;
      }
      break;

    case DTYPE_U16:
      switch (dst->dtype) {
        case DTYPE_U8:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint8_t *)dst->data)[i] = (uint8_t)((uint16_t *)src->data)[i];
          }
          return;
        case DTYPE_I16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int16_t *)dst->data)[i] = (int16_t)((uint16_t *)src->data)[i];
          }
          return;
        case DTYPE_U32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint32_t *)dst->data)[i] = (uint32_t)((uint16_t *)src->data)[i];
          }
          return;
        case DTYPE_I32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int32_t *)dst->data)[i] = (int32_t)((uint16_t *)src->data)[i];
          }
          return;
        case DTYPE_F32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((float *)dst->data)[i] = (float)((uint16_t *)src->data)[i];
          }
          return;
        default:
          break;
      }
      break;
  
    case DTYPE_I16:
      switch (dst->dtype) {
        case DTYPE_U8:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint8_t *)dst->data)[i] = (uint8_t)((int16_t *)src->data)[i];
          }
          return;
        case DTYPE_I8:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int8_t *)dst->data)[i] = (int8_t)((int16_t *)src->data)[i];
          }
          return;
        case DTYPE_U16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint16_t *)dst->data)[i] = (uint16_t)((int16_t *)src->data)[i];
          }
          return;
        case DTYPE_U32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint32_t *)dst->data)[i] = (uint32_t)((int16_t *)src->data)[i];
          }
          return;
        case DTYPE_I32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int32_t *)dst->data)[i] = (int32_t)((int16_t *)src->data)[i];
          }
          return;
        default:
          break;
      }
      break;
    
    case DTYPE_U32:
      switch (dst->dtype) {
        case DTYPE_U8:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint8_t *)dst->data)[i] = (uint8_t)((uint32_t *)src->data)[i];
          }
          return;
        case DTYPE_I8:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int8_t *)dst->data)[i] = (int8_t)((uint32_t *)src->data)[i];
          }
          return;
        case DTYPE_U16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint16_t *)dst->data)[i] = (uint16_t)((uint32_t *)src->data)[i];
          }
          return;
        case DTYPE_I16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int16_t *)dst->data)[i] = (int16_t)((uint32_t *)src->data)[i];
          }
          return;
        case DTYPE_I32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int32_t *)dst->data)[i] = (int32_t)((uint32_t *)src->data)[i];
          }
          return;
        case DTYPE_F32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((float *)dst->data)[i] = (float)((uint32_t *)src->data)[i];
          }
          return;
        default:
          break;
      }
      break;
  
    case DTYPE_I32:
      switch (dst->dtype) {
        case DTYPE_U8:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint8_t *)dst->data)[i] = (uint8_t)((int32_t *)src->data)[i];
          }
          return;
        case DTYPE_I8:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int8_t *)dst->data)[i] = (int8_t)((int32_t *)src->data)[i];
          }
          return;
        case DTYPE_U16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint16_t *)dst->data)[i] = (uint16_t)((int32_t *)src->data)[i];
          }
          return;
        case DTYPE_I16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int16_t *)dst->data)[i] = (int16_t)((int32_t *)src->data)[i];
          }
          return;
        case DTYPE_U32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((uint32_t *)dst->data)[i] = (uint32_t)((int32_t *)src->data)[i];
          }
          return;
        case DTYPE_F32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((float *)dst->data)[i] = (float)((int32_t *)src->data)[i];
          }
          return;
        default:
          break;
      }
      break;
    
    case DTYPE_F16:
      switch (dst->dtype) {
        case DTYPE_F32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((float *)dst->data)[i] = NN_half_to_float(((float16_t *)src->data)[i]);
          }
          return;
        default:
          break;
      }
      break;
    
    case DTYPE_F32:
      switch (dst->dtype) {
        case DTYPE_I32:
          for (size_t i = 0; i < src->size; i += 1) {
            ((int32_t *)dst->data)[i] = (int32_t)((float *)src->data)[i];
          }
          return;
        case DTYPE_F16:
          for (size_t i = 0; i < src->size; i += 1) {
            ((float16_t *)dst->data)[i] = NN_float_to_half(((float *)src->data)[i]);
          }
          return;
        default:
          break;
      }
      break;
  }
  printf("[ERROR] Cannot copy tensor from type %s to %s\n", NN_get_datatype_name(src->dtype), NN_get_datatype_name(dst->dtype));
}
