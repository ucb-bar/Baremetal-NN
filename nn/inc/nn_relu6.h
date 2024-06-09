#ifndef __NN_RELU6_H
#define __NN_RELU6_H

#include <assert.h>

#include "nn_tensor.h"

/**
 * Applies the ReLU6 function element-wise.
 * 
 * y = ReLU6(x) = min(max(0, x), 6)
 * 
 * @param y: the output tensor
 * @param x: the input tensor
 */
void NN_ReLU6_F32(Tensor *y, Tensor *x);

void NN_ReLU6Inplace_F32(Tensor *x);


#endif // __NN_RELU_H
