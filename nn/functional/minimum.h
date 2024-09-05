#ifndef __NN_MINIMUM_H
#define __NN_MINIMUM_H

#include <assert.h>

#include "tensor.h"
#include "kernel/minimum.h"


/**
 * Computes the element-wise minimum of two tensors.
 * 
 * @param out: the output tensor
 * @param a: the input tensor
 * @param b: the input tensor
 */
void NN_minimum(Tensor *out, const Tensor *a, const Tensor *b);


#endif // __NN_MINIMUM_H
