#ifndef __NN_conv2d_H
#define __NN_conv2d_H

#include <assert.h>
#include <string.h>

#include "tensor.h"

/**
 * Converts a tensor from NCHW (PyTorch) to NHWC (Channel-last) format.
 * 
 * @param out: the output tensor of shape (batch_size, height, width, channels)
 * @param in: the input tensor of shape (batch_size, channels, height, width)
 */
void NN_nchw_to_nhwc(Tensor *out, const Tensor *in);

/**
 * Converts a tensor from NHWC (Channel-last) to NCHW (PyTorch) format.
 * 
 * @param out: the output tensor of shape (batch_size, channels, height, width)
 * @param in: the input tensor of shape (batch_size, height, width, channels)
 */
void NN_nhwc_to_nchw(Tensor *out, const Tensor *in);

/**
 * Applies a 2D convolution over an input signal composed of several input planes.
 * 
 * @param out: the output tensor of shape (batch_size, channels_in, height, width)
 * @param in: the input tensor of shape (batch_size, channels_out, height, width)
 * @param weight: the learnable weights of the module of shape (kernel_height, kernel_width, channels_in, channels_out)
 * @param bias: the learnable bias of the module of shape (channels_out), or NULL if no bias is applied
 * @param stride: stride for the cross-correlation
 * @param padding: the amount of padding applied to the input
 * @param dilation: the spacing between kernel elements
 * @param groups: number of blocked connections from input channels to output channels
 */
void NN_conv2d(
  Tensor *out, const Tensor *in,
  const Tensor *weight, const Tensor *bias,
  const size_t *stride, const size_t *padding, const size_t *dilation, size_t groups
  );


#endif // __NN_conv2d_H
