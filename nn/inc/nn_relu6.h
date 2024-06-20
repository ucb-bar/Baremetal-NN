#ifndef __NN_RELU6_H
#define __NN_RELU6_H

#include <assert.h>

#include "nn_tensor.h"
#include "ops/maximum1.h"
#include "ops/minimum1.h"


/**
 * Applies the ReLU6 function element-wise.
 * 
 * y = ReLU6(x) = min(max(x, 0), 6)
 * 
 * @param y: the output tensor
 * @param x: the input tensor
 */
void NN_ReLU6(Tensor *y, Tensor *x);

void NN_ReLU6Inplace(Tensor *x);


#endif // __NN_RELU6_H
