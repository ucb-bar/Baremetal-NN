#ifndef __NN_MAXPOOL2D_H
#define __NN_MAXPOOL2D_H

#include <assert.h>

#include "nn_tensor.h"
#include "nn_max.h"


/**
 * Applies a 2D max pooling over an input signal composed of several input planes.
 * 
 * @param out - output tensor
 * @param in - input tensor
 * @param kernel_size - size of the pooling window
 * @param stride - stride of the pooling window
 */
void NN_maxpool2d_F32(Tensor *out, Tensor *in, size_t *kernel_size, size_t *stride);

#endif // __NN_MAXPOOL2D_H
