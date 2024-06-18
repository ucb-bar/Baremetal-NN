#ifndef __NN_RELU_H
#define __NN_RELU_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"


/**
 * Applies the Exponential Linear Unit (ELU) function, element-wise.
 * 
 * The ELU function is defined as:
 * 
 *    ELU(x) = x, if x > 0
 *             alpha * (exp(x) - 1), if x <= 0
 * 
 * @param y: output tensor
 * @param x: input tensor
 * @param alpha: the alpha value for the ELU formulation
 */
void NN_ELU(Tensor *y, Tensor *x, float alpha);

void NN_ELUInplace(Tensor *x, float alpha);


#endif // __NN_RELU_H
