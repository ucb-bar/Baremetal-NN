#ifndef __NN_MUL_H
#define __NN_MUL_H

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
void NN_mul(Tensor *out, Tensor *in, float scalar);

void NN_mul_F32(Tensor *out, Tensor *in, float scalar);

void NN_mul_F32_RVV(Tensor *out, Tensor *in, float scalar);


#endif // __NN_MUL_H
