
#include "nn_conv2d.h"


void NN_Conv2d_F32(
  Tensor *out, const Tensor *in, 
  const Tensor *weight, const Tensor *bias, 
  const size_t *stride, const size_t *padding, size_t groups) {
  const size_t dilation[2] = {1, 1};

  assert(in->ndim == 4);
  assert(out->ndim == 4);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(weight->dtype == DTYPE_F32);
  if (bias != NULL) { 
    assert(bias->ndim == 1);
    assert(bias->dtype == DTYPE_F32);
  }
  assert(out->shape[0] == in->shape[0]);
  assert(out->shape[1] == weight->shape[0]);
  assert(in->shape[1] == weight->shape[1] * groups);
  assert(out->shape[2] == (in->shape[2] + 2 * padding[0] - dilation[0] * (weight->shape[2] - 1) - 1) / stride[0] + 1);
  assert(out->shape[3] == (in->shape[3] + 2 * padding[1] - dilation[1] * (weight->shape[3] - 1) - 1) / stride[1] + 1);
  assert(groups > 0);
  assert(in->shape[1] % groups == 0);
  assert(out->shape[1] % groups == 0);

  size_t batch_size = in->shape[0];
  size_t out_channels = out->shape[1];
  size_t in_channels = in->shape[1];
  size_t input_height = in->shape[2];
  size_t input_width = in->shape[3];
  size_t kernel_height = weight->shape[2];
  size_t kernel_width = weight->shape[3];
  size_t stride_height = stride[0];
  size_t stride_width = stride[1];
  size_t padding_height = padding[0];
  size_t padding_width = padding[1];

  size_t output_height = (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
  size_t output_width = (input_width + 2 * padding_width - kernel_width) / stride_width + 1;

  // Initialize output tensor to zeros
  memset(out->data, 0, batch_size * out_channels * output_height * output_width * sizeof(float));

  for (size_t n = 0; n < batch_size; n += 1) {
    for (size_t g = 0; g < groups; g += 1) {
      for (size_t oc = 0; oc < out_channels / groups; oc += 1) {
        for (size_t oh = 0; oh < output_height; oh += 1) {
          for (size_t ow = 0; ow < output_width; ow += 1) {
            float sum = 0.f;
            if (bias != NULL) {
              sum = ((float *)bias->data)[g * (out_channels / groups) + oc];
            }
            for (size_t ic = 0; ic < in_channels / groups; ic += 1) {
              for (size_t kh = 0; kh < kernel_height; kh += 1) {
                for (size_t kw = 0; kw < kernel_width; kw += 1) {
                  size_t ih = oh * stride_height + kh - padding_height;
                  size_t iw = ow * stride_width + kw - padding_width;
                  if (ih < input_height && iw < input_width) {
                    size_t in_idx = n * in_channels * input_height * input_width
                                  + (g * (in_channels / groups) + ic) * input_height * input_width
                                  + ih * input_width
                                  + iw;
                    size_t weight_idx = g * (out_channels / groups) * (in_channels / groups) * kernel_height * kernel_width
                                        + oc * (in_channels / groups) * kernel_height * kernel_width
                                        + ic * kernel_height * kernel_width
                                        + kh * kernel_width
                                        + kw;

                    sum += ((float *)in->data)[in_idx] * ((float *)weight->data)[weight_idx];
                  }
                }
              }
            }
            size_t out_idx = n * out_channels * output_height * output_width
                               + (g * (out_channels / groups) + oc) * output_height * output_width
                               + oh * output_width
                               + ow;
            ((float *)out->data)[out_idx] = sum;
          }
        }
      }
    }
  }
}
