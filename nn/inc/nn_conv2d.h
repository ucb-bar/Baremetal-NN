#ifndef __NN_CONV2D_H
#define __NN_CONV2D_H

#include <assert.h>

#include "nn_tensor.h"


/**
 * Applies a 2D convolution over an input signal composed of several input planes.
 * 
 * @param out: the output tensor of shape (batch_size, channels_in, height, width)
 * @param in: the input tensor of shape (batch_size, channels_out, height, width)
 * @param weight: the learnable weights of the module of shape (channels_out, channels_in, kernel_size[0], kernel_size[1])
 * @param bias: the learnable bias of the module of shape (channels_out), or NULL if no bias is applied
 * @param in_channels: number of channels in the input tensor
 * @param out_channels: number of channels produced by the convolution
 * @param kernel_size: size of the convolution kernel
 * @param stride: stride for the cross-correlation
 * @param padding: the amount of padding applied to the input
 * @param groups: number of blocked connections from input channels to output channels
 */
void NN_Conv2d_F32(
  Tensor *out, Tensor *in, 
  Tensor *weight, Tensor *bias, 
  size_t *kernel_size, size_t *stride, size_t *padding, size_t groups
  );


#endif // __NN_CONV2D_H
