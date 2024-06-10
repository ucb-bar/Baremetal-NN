

#include "nn_maxpool2d.h"

void NN_MaxPool2d_F32(Tensor *out, Tensor *in, size_t *kernel_size) {
  assert(in->ndim == 4);
  assert(out->ndim == 4);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->shape[0] == in->shape[0]);
  assert(out->shape[1] == in->shape[1]);
  assert(in->shape[2] >= kernel_size[0]);
  assert(in->shape[3] >= kernel_size[1]);
  assert(out->shape[2] == in->shape[2] - kernel_size[0] + 1);
  assert(out->shape[3] == in->shape[3] - kernel_size[0] + 1);

  // input (batch_size, channels, height, width)
  // output (batch_size, channels, pooled_height, pooled_width)

  size_t batch_size = in->shape[0];
  size_t channels = in->shape[1];
  size_t input_height = in->shape[2];
  size_t input_width = in->shape[3];

  size_t kernel_height = kernel_size[0];
  size_t kernel_width = kernel_size[1];

  size_t output_height = (input_height - kernel_height) + 1;
  size_t output_width = (input_width - kernel_width) + 1;

  size_t stride = kernel_size[0];

  for (size_t b = 0; b < batch_size; b += 1) {
    for (size_t c = 0; c < channels; c += 1) {
      for (size_t h = 0; h < output_height; h += 1) {
        for (size_t w = 0; w < output_width; w += 1) {
          // Calculate the offset of the current pooling window
          size_t window_offset = b * in->shape[1] * in->shape[2] * in->shape[3]
                               + c * in->shape[2] * in->shape[3] 
                               + h * in->shape[3]
                               + w;

          // Create a tensor for the current pooling window
          Tensor *window = NN_tensor(2, (size_t[]){kernel_height, kernel_width}, DTYPE_F32, ((float *)in->data) + window_offset);

          *(((float *)out->data) + window_offset) = NN_max(window);
        }
      }
    }
  }
}
