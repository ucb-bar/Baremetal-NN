#ifndef __NN_MUL_H
#define __NN_MUL_H

#include <assert.h>
#include <float.h>

#include "nn_tensor.h"


/**
 * Returns the element-wise multiplication of two tensors.
 * 
 * out_i = a_i * b_i
 * 
 * @param out: the output tensor
 * @param a: the input tensor
 * @param b: the input tensor
 */
void NN_mul(Tensor *out, Tensor *a, Tensor *b);

void NN_mul_F32(Tensor *out, Tensor *a, Tensor *b);


void NN_mul_F32_RVV(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_MUL_H
