#ifndef __NN_RELU_H
#define __NN_RELU_H

#include <assert.h>

#include "nn_tensor.h"
#include "ops/maximum1.h"


/**
 * Applies the rectified linear unit function element-wise
 * 
 * y = ReLU(x) = max(x, 0)
 * 
 * @param y: the output tensor
 * @param x: the input tensor
 */
void NN_ReLU(Tensor *y, Tensor *x);

void NN_ReLUInplace(Tensor *x);

#endif // __NN_RELU_H
