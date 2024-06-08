

#include "nn_maxpool2d.h"

void NN_maxpool2d_F32(Tensor *out, Tensor *in, size_t *kernel_size, size_t *stride) {
  // Input tensor should be 4D: (batch_size, channels, height, width)
  assert(in->ndim == 4);
  assert(in->dtype == DTYPE_F32);
  
  // Output tensor should be 4D: (batch_size, channels, pooled_height, pooled_width)
  assert(out->ndim == 4);
  assert(out->dtype == DTYPE_F32);

  float pool_height = kernel_size[0];
  float pool_width = kernel_size[1];
  float stride_height = stride[0];
  float stride_width = stride[1];

  int batch_size = in->shape[0];
  int channels = in->shape[1];
  int input_height = in->shape[2];
  int input_width = in->shape[3];
  int output_height = (input_height - pool_height) / stride_height + 1;
  int output_width = (input_width - pool_width) / stride_width + 1;
  
  float *in_ptr = (float *)in->data;
  float *out_ptr = (float *)out->data;
  
  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
          // Find the starting point for the current pooling window
          int h_start = h * stride_height;
          int w_start = w * stride_width;
          // Find the ending point for the current pooling window
          int h_end = h_start + pool_height;
          int w_end = w_start + pool_width;

          // Create a tensor for the current pooling window
          Tensor window;
          window.ndim = 2;
          window.shape[0] = pool_height;
          window.shape[1] = pool_width;
          window.dtype = DTYPE_F32;
          window.data = in_ptr + n * in->shape[1] * in->shape[2] * in->shape[3] + c * in->shape[2] * in->shape[3] + h_start * in->shape[3] + w_start;

          // Get the maximum value in the current pooling window
          float max_value = NN_max(&window);

          // Store the maximum value in the output tensor
          out_ptr[n * out->shape[1] * out->shape[2] * out->shape[3] + c * out->shape[2] * out->shape[3] + h * out->shape[3] + w] = max_value;
        }
      }
    }
  }
}
