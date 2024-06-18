
#include "nn_conv2d.h"
#include "gemmini/gemmini.h"

void NN_Conv2dNHWC_F32_Gemmini(
  Tensor *out, Tensor *in, 
  Tensor *weight, Tensor *bias, 
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
  assert(out->shape[3] == weight->shape[0]);
  assert(in->shape[3] == weight->shape[2] * groups);
  assert(out->shape[1] == (in->shape[0] + 2 * padding[0] - dilation[0] * (weight->shape[0] - 1) - 1) / stride[0] + 1);
  assert(out->shape[2] == (in->shape[1] + 2 * padding[1] - dilation[1] * (weight->shape[1] - 1) - 1) / stride[1] + 1);
  assert(groups > 0);
  assert(in->shape[3] % groups == 0);
  assert(out->shape[3] % groups == 0);

  size_t batch_size = in->shape[0];
  size_t out_channels = out->shape[1];
  size_t in_channels = in->shape[1];
  size_t in_height = in->shape[2];
  size_t in_width = in->shape[3];
  size_t kernel_height = weight->shape[2];
  size_t kernel_width = weight->shape[3];
  size_t stride_height = stride[0];
  size_t stride_width = stride[1];
  size_t padding_height = padding[0];
  size_t padding_width = padding[1];

  size_t out_height = (in_height + 2 * padding_height - kernel_height) / stride_height + 1;
  size_t out_width = (in_width + 2 * padding_width - kernel_width) / stride_width + 1;

  int in_stride = in_channels;
  int out_stride = out_channels;
  int weight_stride = out_channels;
  tiled_conv_stride_auto(
    batch_size, in_height, in_width, in_channels,
    out_channels, out_height, out_width,
    stride_height, dilation[0], dilation[0], padding_height, kernel_height,
    in_stride, weight_stride, out_stride,
    0, 0, 0,
    0, 0,
    
    in->data, weight->data, NULL, out->data,

    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, WS);
  
  
}
