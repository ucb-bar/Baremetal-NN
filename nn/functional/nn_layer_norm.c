
#include "nn_layer_norm.h"

void NN_layer_norm(
  Tensor *out, Tensor *in,
  size_t normalized_dims,
  Tensor *weight, Tensor *bias,
  const float eps) {
  assert(out->dtype == in->dtype && in->dtype == DTYPE_F32);
  assert(out->ndim == in->ndim);

  // currently only support 1D normalization
  assert(normalized_dims == 1);

  size_t n = in->shape[1];
  for (size_t i = 0; i < in->shape[0]; i += 1) {
    float *out_ptr = (float *)out->data + i * n;
    float *in_ptr = (float *)in->data + i * n;
    
    float mean = 0;
    NN__sum_f32(n, &mean, in_ptr, 1);
    mean /= n;

    float variance = 0;
    // use y as temporary buffer
    // y = x - E[x]
    NN__add1_f32(n, out_ptr, 1, in_ptr, 1, -mean);
    // y = y * y
    NN__sqr_f32(n, out_ptr, 1, out_ptr, 1);

    NN__sum_f32(n, &variance, out_ptr, 1);
    variance /= n;

    // y = x - E[x]
    NN__add1_f32(n, out_ptr, 1, in_ptr, 1, -mean);

    // y = y / sqrt(Var[x] + eps)
    NN__mul1_f32(n, out_ptr, 1, out_ptr, 1, 1.f / sqrtf(variance + eps));

    // y = y * weight + bias
    NN__mul_f32(n, out_ptr, 1, (float *)weight->data, 1, out_ptr, 1);
    NN__add_f32(n, out_ptr, 1, (float *)bias->data, 1, out_ptr, 1);
  }
}
