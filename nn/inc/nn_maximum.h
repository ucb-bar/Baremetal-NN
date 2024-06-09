#ifndef __NN_MAXIMUM_H
#define __NN_MAXIMUM_H

#include <assert.h>
#include <float.h>

#include "nn_tensor.h"


/**
 * Computes the element-wise maximum of two tensors.
 * 
 * @param out: the output tensor
 * @param a: the input tensor
 * @param b: the input tensor
 */
void NN_maximum(Tensor *out, Tensor *a, Tensor *b);

void NN_maximum_F32(Tensor *out, Tensor *a, Tensor *b);


void NN_maximum_F32_RVV(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_MAXIMUM_H
