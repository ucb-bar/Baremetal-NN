#ifndef __NN_RELU_H
#define __NN_RELU_H

#include <assert.h>

#include "nn_types.h"
#include "nn_add.h"
#include "nn_matmul.h"


/**
 * Applies the rectified linear unit function element-wise: y = max(0, x)
 * 
 */
void NN_relu_F32(Tensor *y, Tensor *x);

#endif // __NN_RELU_H
