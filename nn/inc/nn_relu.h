#ifndef __NN_RELU_H
#define __NN_RELU_H

#include <assert.h>

#include "nn_tensor.h"

/**
 * Applies the rectified linear unit function element-wise
 * 
 * y = max(0, x)
 * 
 */
void NN_ReLU_F32(Tensor *y, Tensor *x);

void NN_ReLUInplace_F32(Tensor *x);


#endif // __NN_RELU_H
