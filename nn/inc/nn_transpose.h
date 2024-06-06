#ifndef __NN_TRANSPOSE_H
#define __NN_TRANSPOSE_H

#include <assert.h>

#include "nn_tensor.h"


/**
 * Transpose a 2D tensor
 * 
 * @warning this is not an in-place operation, the output tensor should be different from the input tensor
 * 
 * @param out: output tensor of shape (n, m)
 * @param a: input tensor of shape (m, n)
 */
void NN_transpose(Tensor *out, Tensor *a);

void NN_transpose_I8(Tensor *out, Tensor *a);

void NN_transpose_I32(Tensor *out, Tensor *a);

void NN_transpose_F32(Tensor *out, Tensor *a);


#endif // __NN_TRANSPOSE_H
