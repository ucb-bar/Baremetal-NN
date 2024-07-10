#ifndef __NN_ELU_H
#define __NN_ELU_H

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
void NN_elu(Tensor *y, Tensor *x, float alpha);

void NN_elu_inplace(Tensor *x, float alpha);


#endif // __NN_ELU_H
