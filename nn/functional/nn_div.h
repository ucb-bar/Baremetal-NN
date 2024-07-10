#ifndef __NN_DIV_H
#define __NN_DIV_H

#include <assert.h>
#include <float.h>

#include "nn_tensor.h"
#include "div.h"


/**
 * Returns the element-wise multiplication of two tensors.
 * 
 * out_i = a_i / b_i
 * 
 * @param out: the output tensor
 * @param a: the input tensor
 * @param b: the input tensor
 */
void NN_div(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_DIV_H
