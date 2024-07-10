
#include "nn_layer_norm.h"

void NN_layer_norm(
  Tensor *out, Tensor *in,
  Tensor *weight, Tensor *bias,
  const float eps) {
  assert(out->dtype == in->dtype && in->dtype == DTYPE_F32);
  assert(out->ndim == in->ndim);

  size_t N = in->shape[1];
  for (size_t i = 0; i < in->shape[0]; i++) {
    float mean = 0;
    for (size_t j = 0; j < N; j++) {
      mean += ((float *)in->data)[i * N + j];
    }
    mean /= N;

    float variance = 0;
    for (size_t j = 0; j < N; j++) {
      variance += powf(((float *)in->data)[i * N + j] - mean, 2);
    }
    variance /= N;

    for (size_t j = 0; j < N; j++) {
      ((float *)out->data)[i * N + j] = ((float *)weight->data)[j] * (((float *)in->data)[i * N + j] - mean) / sqrtf(variance + eps) + ((float *)bias)[j];
    }
  }
}
