#ifndef __NN_MATRIXNORM_H
#define __NN_MATRIXNORM_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"


/**
 * Computes the Frobenius norm of a matrix.
 * 
 * @param tensor: the input tensor of shape (m, n)
 */
void NN_matrixNorm(Tensor *scalar, Tensor *x);

void NN_matrixNorm_F32(Tensor *scalar, Tensor *x);


#endif // __NN_MATRIXNORM_H
