#ifndef __NN_BATCH_NORM2D_H
#define __NN_BATCH_NORM2D_H

#include <assert.h>
#include <math.h>

#include "tensor.h"


/**
 * Applies Batch Normalization over a 4D input.
 * 
 * @param out: the output tensor of shape (batch_size, height, width, channels)
 * @param in: the input tensor of shape (batch_size, height, width, channels)
 * @param weight: the learnable weights of the module of shape (channels), or NULL if no weight is applied
 * @param bias: the learnable bias of the module of shape (channels), or NULL if no bias is applied
 * @param eps: a value added to the denominator for numerical stability
 * @param running_mean: the running mean of the module of shape (channels), or NULL if no running mean is applied
 * @param running_var: the running variance of the module of shape (channels), or NULL if no running variance is applied
 */
void NN_batch_norm2d(
    Tensor *out, const Tensor *in,
    const Tensor *weight, const Tensor *bias,
    float eps, const Tensor *running_mean, const Tensor *running_va
    );


#endif // __NN_BATCH_NORM2D_H
