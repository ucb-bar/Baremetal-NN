#ifndef __NN_MAXPOOL2D_H
#define __NN_MAXPOOL2D_H

#include <assert.h>

#include "nn_tensor.h"
#include "nn_max.h"


/**
 * Applies a 2D max pooling over an input signal composed of several input planes.
 * 
 * @param out - output tensor of shape (batch_size, channels, height, width)
 * @param in - input tensor of shape (batch_size, channels, pooled_height, pooled_width)
 * @param kernel_size - size of the pooling window
 */
void NN_MaxPool2d_F32(Tensor *out, Tensor *in, size_t *kernel_size);


#endif // __NN_MAXPOOL2D_H
