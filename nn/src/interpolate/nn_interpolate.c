
#include "nn_interpolate.h"

void NN_interpolate_F32(Tensor *out, Tensor *in, float scale_factor/*const char* mode*/) {
  assert(in->ndim == 4);
  assert(out->ndim == 4);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(scale_factor > 0);

  size_t batch_size = in->shape[0];
  size_t channels = in->shape[1];
  size_t input_height = in->shape[2];
  size_t input_width = in->shape[3];
  size_t output_height = out->shape[2];
  size_t output_width = out->shape[3];

  // Ensure output dimensions match the expected dimensions after scaling
  assert(output_height == (size_t)(input_height * scale_factor));
  assert(output_width == (size_t)(input_width * scale_factor));

  // Initialize output tensor to zeros
  memset(out->data, 0, batch_size * channels * output_height * output_width * sizeof(float));

    for (size_t n = 0; n < batch_size; n++) {
      for (size_t c = 0; c < channels; c++) {
        for (size_t oh = 0; oh < output_height; oh++) {
          for (size_t ow = 0; ow < output_width; ow++) {
            size_t ih = (size_t)(oh / scale_factor);
            size_t iw = (size_t)(ow / scale_factor);

            size_t in_idx = n * channels * input_height * input_width
                          + c * input_height * input_width
                          + ih * input_width
                          + iw;
            size_t out_idx = n * channels * output_height * output_width
                           + c * output_height * output_width
                           + oh * output_width
                           + ow;

            ((float *)out->data)[out_idx] = ((float *)in->data)[in_idx];
          }
        }
      }
    }
}
