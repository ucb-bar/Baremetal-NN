
#include "nn_interpolate.h"

void NN_interpolate(Tensor *out, Tensor *in, const float *scale_factor/*const char* mode*/) {
  assert(in->ndim == 4);
  assert(out->ndim == 4);
  assert(scale_factor[0] > 0);
  assert(scale_factor[1] > 0);

  size_t batch_size = in->shape[0];
  size_t input_height = in->shape[1];
  size_t input_width = in->shape[2];
  size_t channels = in->shape[3];
  size_t output_height = out->shape[1];
  size_t output_width = out->shape[2];

  // Ensure output dimensions match the expected dimensions after scaling
  assert(output_height == (size_t)(input_height * scale_factor[0]));
  assert(output_width == (size_t)(input_width * scale_factor[1]));

  // Initialize output tensor to zeros
  memset(out->data, 0, batch_size * channels * output_height * output_width * NN_sizeof(out->dtype));

  for (size_t n = 0; n < batch_size; n += 1) {
    for (size_t oh = 0; oh < output_height; oh += 1) {
      for (size_t ow = 0; ow < output_width; ow += 1) {
        for (size_t c = 0; c < channels; c += 1) {
          size_t ih = (size_t)(oh / scale_factor[0]);
          size_t iw = (size_t)(ow / scale_factor[1]);

          size_t in_idx = n * input_height * input_width * channels
                        + ih * input_width * channels
                        + iw * channels + c;
          size_t out_idx = n * output_height * output_width * channels
                        + oh * output_width * channels
                        + ow * channels + c;

          switch (out->dtype) {
            case DTYPE_U8:
              ((uint8_t *)out->data)[out_idx] = ((uint8_t *)in->data)[in_idx];
              break;

            case DTYPE_F32:
              ((float *)out->data)[out_idx] = ((float *)in->data)[in_idx];
              break;

            default:
              printf("[ERROR] Unsupported operation of tensor with dtype %s = |%s|\n", 
                NN_get_datatype_name(out->dtype), NN_get_datatype_name(in->dtype)
              );
              break;
          }
        }
      }
    }
  }
}
