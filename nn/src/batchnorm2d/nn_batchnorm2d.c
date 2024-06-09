
#include "nn_batchnorm2d.h"

void NN_BatchNorm2d_F32(
  Tensor *out, Tensor *in, 
  Tensor *weight, Tensor *bias,
  float eps, float momentum, Tensor *running_mean, Tensor *running_var) {
  assert(in->ndim == 4);
  assert(out->ndim == 4);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(weight->dtype == DTYPE_F32);
  assert(bias->dtype == DTYPE_F32);
  assert(running_mean->dtype == DTYPE_F32);
  assert(running_var->dtype == DTYPE_F32);

  size_t batch_size = in->shape[0];
  size_t channels = in->shape[1];
  size_t height = in->shape[2];
  size_t width = in->shape[3];

  // Temporary tensors to hold mean and variance for each channel
  float *mean = (float *)malloc(channels * sizeof(float));
  float *var = (float *)malloc(channels * sizeof(float));

  // Calculate mean
  for (size_t c = 0; c < channels; c += 1) {
    mean[c] = 0.0;
    for (size_t b = 0; b < batch_size; b += 1) {
      for (size_t h = 0; h < height; h += 1) {
        for (size_t w = 0; w < width; w += 1) {
          size_t index = b * channels * height * width + c * height * width + h * width + w;
          mean[c] += ((float *)in->data)[index];
        }
      }
    }
    mean[c] /= (batch_size * height * width);
  }

  // Calculate variance
  for (size_t c = 0; c < channels; c += 1) {
    var[c] = 0.0;
    for (size_t b = 0; b < batch_size; b += 1) {
      for (size_t h = 0; h < height; h += 1) {
        for (size_t w = 0; w < width; w += 1) {
          size_t index = b * channels * height * width + c * height * width + h * width + w;
          var[c] += pow(((float *)in->data)[index] - mean[c], 2);
        }
      }
    }
    var[c] /= (batch_size * height * width);
  }

  // Update running mean and variance
  for (size_t c = 0; c < channels; c += 1) {
    ((float *)running_mean->data)[c] = momentum * ((float *)running_mean->data)[c] + (1 - momentum) * mean[c];
    ((float *)running_var->data)[c] = momentum * ((float *)running_var->data)[c] + (1 - momentum) * var[c];
  }

  // Normalize, scale, and shift
  for (size_t b = 0; b < batch_size; b += 1) {
    for (size_t c = 0; c < channels; c += 1) {
      for (size_t h = 0; h < height; h += 1) {
        for (size_t w = 0; w < width; w += 1) {
          size_t index = b * channels * height * width + c * height * width + h * width + w;
          ((float *)out->data)[index] = ((float *)weight->data)[c] * 
                                        (((float *)in->data)[index] - mean[c]) / 
                                        sqrt(var[c] + eps) + 
                                        ((float *)bias->data)[c];
        }
      }
    }
  }

  free(mean);
  free(var);
}
