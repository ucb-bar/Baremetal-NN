#ifndef __NN_BATCHNORM2D_H
#define __NN_BATCHNORM2D_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"


/**
 * Applies Batch Normalization over a 4D input.
 * 
 * @param out: the output tensor of shape (batch_size, channels, height, width)
 * @param in: the input tensor of shape (batch_size, channels, height, width)
 * @param weight: the learnable weights of the module of shape (channels), or NULL if no weight is applied
 * @param bias: the learnable bias of the module of shape (channels), or NULL if no bias is applied
 * @param eps: a value added to the denominator for numerical stability
 * @param running_mean: the running mean of the module of shape (channels), or NULL if no running mean is applied
 * @param running_var: the running variance of the module of shape (channels), or NULL if no running variance is applied
 */
void NN_BatchNorm2d_F32(
    Tensor *out, Tensor *in,
    Tensor *weight, Tensor *bias,
    float eps, Tensor *running_mean, Tensor *running_va
    );


#endif // __NN_BATCHNORM2D_H
