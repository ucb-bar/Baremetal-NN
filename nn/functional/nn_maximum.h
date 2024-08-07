#ifndef __NN_MAXIMUM_H
#define __NN_MAXIMUM_H

#include <assert.h>

#include "nn_tensor.h"
#include "impl/maximum.h"


/**
 * Computes the element-wise maximum of two tensors.
 * 
 * @param out: the output tensor
 * @param a: the input tensor
 * @param b: the input tensor
 */
void NN_maximum(Tensor *out, const Tensor *a, const Tensor *b);


#endif // __NN_MAXIMUM_H
