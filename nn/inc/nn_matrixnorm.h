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
float NN_matrixNorm_F32(Tensor *tensor);


float NN_matrixNorm_F32_RVV(Tensor *tensor);


#endif // __NN_MATRIXNORM_H
