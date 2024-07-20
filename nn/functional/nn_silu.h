#ifndef __NN_SILU_H
#define __NN_SILU_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"
#include "maximum1.h"


/**
 * Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
 * 
 * The SiLU function is also known as the swish function.
 * 
 * y = silu(x) = x * theta(x), where theta(x) is the logistic sigmoid.
 * 
 * @param y: the output tensor
 * @param x: the input tensor
 */
void NN_silu(Tensor *y, const Tensor *x);

void NN_silu_inplace(Tensor *x);

#endif // __NN_SILU_H
