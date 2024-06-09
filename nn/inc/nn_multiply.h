#ifndef __NN_MULTIPLY_H
#define __NN_MULTIPLY_H

#include <assert.h>
#include <float.h>

#include "nn_tensor.h"


/**
 * Returns the element-wise multiplication of the input tensor with a scalar.
 * 
 * @param out: the output tensor
 * @param in: the input tensor
 * @param scalar: scalar value
 */
void NN_multiply(Tensor *out, Tensor *in, float scalar);

void NN_multiply_F32(Tensor *out, Tensor *in, float scalar);

void NN_multiply_F32_RVV(Tensor *out, Tensor *in, float scalar);


#endif // __NN_MULTIPLY_H
