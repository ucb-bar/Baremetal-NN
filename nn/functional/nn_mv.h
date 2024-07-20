#ifndef __NN_MV_H
#define __NN_MV_H

#include <assert.h>

#include "nn_tensor.h"
#include "impl/dot.h"


/**
 * Performs a matrix multiplication.
 * 
 * If input is a (n, m) tensor, v is a (m, ) tensor, out will be a (n, ) tensor.
 * 
 * C = A @ B
 *
 * @param out: the output tensor of shape (n,)
 * @param a: the input tensor of shape (n, m)
 * @param v: the input vector tensor of shape (m,)
 */
void NN_mv(Tensor *out, const Tensor *a, const Tensor *v);


#endif // __NN_MV_H
