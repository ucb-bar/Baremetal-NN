#ifndef __NN_RELU_H
#define __NN_RELU_H

#include <assert.h>
#include <math.h>

#include "nn_types.h"
#include "nn_add.h"
#include "nn_matmul.h"


/**
 * Applies the rectified linear unit function element-wise: y = max(0, x)
 * 
 * @param y: output tensor
 * @param x: input tensor
 * @param alpha: the alpha value for the ELU formulation
 */
void NN_elu_F32(Tensor *y, Tensor *x, float alpha);


#endif // __NN_RELU_H
