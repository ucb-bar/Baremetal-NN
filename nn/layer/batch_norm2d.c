
#include "batch_norm2d.h"

void NN_batch_norm2d(
  Tensor *out, const Tensor *in, 
  const Tensor *weight, const Tensor *bias,
  float eps, const Tensor *running_mean, const Tensor *running_var) {
  assert(in->ndim == 4);
  assert(out->ndim == 4);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(weight->dtype == DTYPE_F32);
  assert(bias->dtype == DTYPE_F32);
  assert(running_mean->dtype == DTYPE_F32);
  assert(running_var->dtype == DTYPE_F32);

  size_t batch_size = in->shape[0];
  size_t height = in->shape[1];
  size_t width = in->shape[2];
  size_t channels = in->shape[3];

  float *mean;
  float *var;

  // Temporary tensors to hold mean and variance for each channel
  if (running_mean != NULL) {
    mean = running_mean->data;
    var = running_var->data;
  }
  else {
    printf("[ERROR] running_mean is NULL\n");
    
    // mean = malloc(channels * sizeof(float));
    // var = malloc(channels * sizeof(float));

    // // Calculate mean
    // for (size_t c = 0; c < channels; c += 1) {
    //   mean[c] = 0.0;
    //   for (size_t b = 0; b < batch_size; b += 1) {
    //     for (size_t h = 0; h < height; h += 1) {
    //       for (size_t w = 0; w < width; w += 1) {
    //         size_t index = b * channels * height * width + c * height * width + h * width + w;
    //         mean[c] += ((float *)in->data)[index];
    //       }
    //     }
    //   }
    //   mean[c] /= (batch_size * height * width);
    // }

    // // Calculate variance
    // for (size_t c = 0; c < channels; c += 1) {
    //   var[c] = 0.0;
    //   for (size_t b = 0; b < batch_size; b += 1) {
    //     for (size_t h = 0; h < height; h += 1) {
    //       for (size_t w = 0; w < width; w += 1) {
    //         size_t index = b * channels * height * width + c * height * width + h * width + w;
    //         var[c] += pow(((float *)in->data)[index] - mean[c], 2);
    //       }
    //     }
    //   }
    //   var[c] /= (batch_size * height * width);
    // }


    // // Use running mean and variance for inference
    // for (size_t c = 0; c < channels; c += 1) {
    //   mean[c] = ((float *)running_mean->data)[c];
    //   var[c] = ((float *)running_var->data)[c];
    // }
  }

  // Normalize, scale, and shift
  for (size_t b = 0; b < batch_size; b += 1) {
    for (size_t h = 0; h < height; h += 1) {
      for (size_t w = 0; w < width; w += 1) {
        for (size_t c = 0; c < channels; c += 1) {
          size_t index = b * height * width * channels + h * width * channels + w * channels + c;
          ((float *)out->data)[index] = ((((float *)in->data)[index] - mean[c]) / sqrt(var[c] + eps))
                                      * ((float *)weight->data)[c]
                                      + ((float *)bias->data)[c];
        }
      }
    }
  }

  if (running_mean == NULL) {
    free(mean);
    free(var);
  }
}
