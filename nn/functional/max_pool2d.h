#ifndef __NN_MAX_POOL2D_H
#define __NN_MAX_POOL2D_H

#include <assert.h>

#include "tensor.h"
#include "max.h"


/**
 * Applies a 2D max pooling over an input signal composed of several input planes.
 * 
 * @param out: the output tensor of shape (batch_size, channels, height, width)
 * @param in: the input tensor of shape (batch_size, channels, pooled_height, pooled_width)
 * @param kernel_size: size of the pooling window
 */
void NN_max_pool2d(Tensor *out, const Tensor *in, const size_t *kernel_size);


#endif // __NN_MAX_POOL2D_H
