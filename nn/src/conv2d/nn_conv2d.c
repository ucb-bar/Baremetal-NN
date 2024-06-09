
#include "nn_conv2d.h"


void NN_Conv2d_F32(Tensor *out, Tensor *in, size_t in_channels, size_t out_channels, size_t *kernel_size, size_t *stride, size_t *padding) {
  const size_t dilation[2] = {1, 1};


  assert(out->shape[2] == (in->shape[2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1);
  assert(out->shape[3] == (in->shape[3] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1);


}
