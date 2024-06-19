
#include "nn_conv2d.h"

#ifdef GEMMINI
  #include "gemmini/gemmini.h"
#endif

void NN_Conv2d(
  Tensor *out, Tensor *in,
  Tensor *weight, Tensor *bias,
  const size_t *stride, const size_t *padding, const size_t *dilation, size_t groups) {
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
  assert(out->shape[3] == weight->shape[3]);
  assert(in->shape[3] == weight->shape[2] * groups);
  assert(out->shape[1] == (in->shape[1] + 2 * padding[0] - dilation[0] * (weight->shape[0] - 1) - 1) / stride[0] + 1);
  assert(out->shape[2] == (in->shape[2] + 2 * padding[1] - dilation[1] * (weight->shape[1] - 1) - 1) / stride[1] + 1);
  assert(groups > 0);
  assert(in->shape[3] % groups == 0);
  assert(out->shape[3] % groups == 0);

  size_t batch_size = in->shape[0];
  size_t out_channels = out->shape[3];
  size_t in_channels = in->shape[3];
  size_t in_height = in->shape[1];
  size_t in_width = in->shape[2];
  size_t kernel_height = weight->shape[0];
  size_t kernel_width = weight->shape[1];
  size_t stride_height = stride[0];
  size_t stride_width = stride[1];
  size_t padding_height = padding[0];
  size_t padding_width = padding[1];
  size_t dilation_height = dilation[0];
  size_t dilation_width = dilation[1];

  size_t out_height = (in_height + 2 * padding_height - kernel_height) / stride_height + 1;
  size_t out_width = (in_width + 2 * padding_width - kernel_width) / stride_width + 1;

  // Initialize output tensor to zeros
  memset(out->data, 0, batch_size * out_height * out_width * out_channels * sizeof(float));

  #ifdef GEMMINI
    #warning "hi gemmini"
    tiled_conv_auto(
        batch_size, in_height, in_width, in_channels,
        out_channels, out_height, out_width,
        stride_height, dilation_height, 1, padding_height, kernel_height, 
        0, 0, 0, 0, 0,

        in->data,
        weight->data,
        bias->data,
        out->data,

        NO_ACTIVATION, ACC_SCALE_IDENTITY,
        0, 0, 0,

        WS);
  #else
    for (size_t n = 0; n < batch_size; n += 1) {
      for (size_t g = 0; g < groups; g += 1) {
        for (size_t oc = 0; oc < out_channels / groups; oc += 1) {
          for (size_t oh = 0; oh < out_height; oh += 1) {
            for (size_t ow = 0; ow < out_width; ow += 1) {
              float sum = 0.f;
              if (bias != NULL) {
                sum = ((float *)bias->data)[g * (out_channels / groups) + oc];
              }
              for (size_t ic = 0; ic < in_channels / groups; ic += 1) {
                for (size_t kh = 0; kh < kernel_height; kh += 1) {
                  for (size_t kw = 0; kw < kernel_width; kw += 1) {
                    int ih = oh * stride_height + kh - padding_height;
                    int iw = ow * stride_width + kw - padding_width;
                    if (ih < (int)in_height && iw < (int)in_width && ih >= 0 && iw >= 0) {
                      size_t in_idx = n * in_height * in_width * in_channels
                                    + ih * in_width * in_channels
                                    + iw * in_channels
                                    + g * (in_channels / groups) + ic;
                      size_t weight_idx = kh * kernel_width * in_channels * out_channels
                                    + kw * in_channels * out_channels
                                    + ic * out_channels
                                    + g * (out_channels / groups) + oc;
                      sum += ((float *)in->data)[in_idx] * ((float *)weight->data)[weight_idx];
                    }
                  }
                }
              }
              size_t out_idx = n * out_height * out_width * out_channels
                            + oh * out_width * out_channels
                            + ow * out_channels
                            + g * (out_channels / groups) + oc;
              ((float *)out->data)[out_idx] = sum;
            }
          }
        }
      }
    }
  #endif
}
