#ifndef __NN_MAXPOOL2D_H
#define __NN_MAXPOOL2D_H

#include <assert.h>

#include "nn_tensor.h"
#include "nn_max.h"


/**
 * Applies a 2D convolution over an input signal composed of several input planes.
 * 
 * @param out: the output tensor of shape (batch_size, channels_in, height, width)
 * @param in: the input tensor of shape (batch_size, channels_out, height, width)
 * @param in_channels: number of channels in the input tensor
 * @param out_channels: number of channels produced by the convolution
 * @param kernel_size: size of the convolution kernel
 * @param stride: stride for the cross-correlation
 * @param padding: the amount of padding applied to the input
 */
void NN_Conv2d_F32(Tensor *out, Tensor *in, size_t in_channels, size_t out_channels, size_t *kernel_size, size_t *stride, size_t *padding);


#endif // __NN_MAXPOOL2D_H
