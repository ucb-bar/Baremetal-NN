
#include "nn_tensor.h"


void NN_as_type(Tensor *out, Tensor *in) {
  if (out->dtype == in->dtype) {
    NN_copy(out, in);
    return;
  }

  switch (in->dtype) {
    case DTYPE_U8:
      switch (out->dtype) {
        case DTYPE_U16:
          for (size_t i = 0; i < in->size; i += 1) {
            ((uint16_t *)out->data)[i] = (uint16_t)((uint8_t *)in->data)[i];
          }
          return;
        case DTYPE_U32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((uint32_t *)out->data)[i] = (uint32_t)((uint8_t *)in->data)[i];
          }
          return;
        case DTYPE_I32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((int32_t *)out->data)[i] = (int32_t)((uint8_t *)in->data)[i];
          }
          return;
      }
      break;
    
    case DTYPE_I8:
      switch (out->dtype) {
        case DTYPE_I16:
          for (size_t i = 0; i < in->size; i += 1) {
            ((int16_t *)out->data)[i] = (int16_t)((int8_t *)in->data)[i];
          }
          return;
        case DTYPE_I32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((int32_t *)out->data)[i] = (int32_t)((int8_t *)in->data)[i];
          }
          return;
        case DTYPE_F32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((float *)out->data)[i] = (float)((int8_t *)in->data)[i];
          }
          return;
      }
      break;
  
    case DTYPE_I16:
      switch (out->dtype) {
        case DTYPE_I32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((int32_t *)out->data)[i] = (int32_t)((int16_t *)in->data)[i];
          }
          return;
      }
      break;
  
    case DTYPE_I32:
      switch (out->dtype) {
        case DTYPE_I8:
          for (size_t i = 0; i < in->size; i += 1) {
            ((int8_t *)out->data)[i] = (int8_t)((int32_t *)in->data)[i];
          }
          return;
        case DTYPE_F32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((float *)out->data)[i] = (float)((int32_t *)in->data)[i];
          }
          return;
      }
      break;
    
    case DTYPE_F16:
      switch (out->dtype) {
        case DTYPE_F32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((float *)out->data)[i] = NN_half_to_float(((float16_t *)in->data)[i]);
          }
          return;
      }
      break;
    
    case DTYPE_F32:
      switch (out->dtype) {
        case DTYPE_I32:
          for (size_t i = 0; i < in->size; i += 1) {
            ((int32_t *)out->data)[i] = (int32_t)((float *)in->data)[i];
          }
          return;
        case DTYPE_F16:
          for (size_t i = 0; i < in->size; i += 1) {
            ((float16_t *)out->data)[i] = NN_float_to_half(((float *)in->data)[i]);
          }
          return;
      }
      break;
  }
  printf("[ERROR] Cannot convert data type from %s to %s\n", NN_get_datatype_name(in->dtype), NN_get_datatype_name(out->dtype));
}
