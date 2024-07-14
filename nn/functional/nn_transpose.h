#ifndef __NN_TRANSPOSE_H
#define __NN_TRANSPOSE_H

#include <assert.h>

#include "nn_tensor.h"
#include "transpose.h"

/**
 * Transpose a 2D tensor
 * 
 * @warning this is not an in-place operation, the output tensor should be different from the input tensor
 * 
 * @param out: the output tensor of shape (n, m)
 * @param a: the input tensor of shape (m, n)
 */
void NN_transpose(Tensor *out, const Tensor *a);


#endif // __NN_TRANSPOSE_H
