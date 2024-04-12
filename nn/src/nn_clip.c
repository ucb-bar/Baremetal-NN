
#include "nn_clip.h"


void NN_clip(Tensor *t, float min, float max) {
  if (t->dtype == DTYPE_F32) {
    return NN_clip_F32(t, min, max);
  }
  if (t->dtype == DTYPE_I8) {
    return NN_clip_I8(t, (int8_t)min, (int8_t)max);
  }
  if (t->dtype == DTYPE_I32) {
    return NN_clip_I32(t, (int32_t)min, (int32_t)max);
  }
  printf("Unsupported data type\n");
}

void NN_clip_I8(Tensor *t, int8_t min, int8_t max) {
  for (size_t i = 0; i<t->size; i+=1) {
    ((int8_t *)t->data)[i] = ((int8_t *)t->data)[i] < min ? min : (((int8_t *)t->data)[i] > max ? max : ((int8_t *)t->data)[i]);
  }
}

void NN_clip_I32(Tensor *t, int32_t min, int32_t max) {
  for (size_t i = 0; i<t->size; i+=1) {
    ((int32_t *)t->data)[i] = ((int32_t *)t->data)[i] < min ? min : (((int32_t *)t->data)[i] > max ? max : ((int32_t *)t->data)[i]);
  }
}

void NN_clip_F32(Tensor *t, float min, float max) {
  for (size_t i = 0; i<t->size; i+=1) {
    ((float *)t->data)[i] = ((float *)t->data)[i] < min ? min : (((float *)t->data)[i] > max ? max : ((float *)t->data)[i]);
  }
}
