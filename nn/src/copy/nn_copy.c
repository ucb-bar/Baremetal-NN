
#include "nn_copy.h"

void NN_copy(Tensor *dst, Tensor *src) {
  assert(dst->ndim == src->ndim);
  assert(dst->dtype == src->dtype);
  assert(dst->shape[0] == src->shape[0]);
  assert(dst->shape[1] == src->shape[1]);
  
  switch (dst->dtype) {
    case DTYPE_I8:
      for (size_t i = 0; i<dst->size; i+=1) {
        ((int8_t *)dst->data)[i] = ((int8_t *)src->data)[i];
      }
      break;
    case DTYPE_I32:
      for (size_t i = 0; i<dst->size; i+=1) {
        ((int32_t *)dst->data)[i] = ((int32_t *)src->data)[i];
      }
      break;
    case DTYPE_F32:
      for (size_t i = 0; i<dst->size; i+=1) {
        ((float *)dst->data)[i] = ((float *)src->data)[i];
      }
      break;
    default:
      printf("[ERROR] Unsupported data type: %d\n", dst->dtype);
  }
}
