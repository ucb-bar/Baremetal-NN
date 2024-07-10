#ifndef __NN_MATMUL_H
#define __NN_MATMUL_H

#include <assert.h>

#include "nn_tensor.h"
#include "dot.h"


/**
 * Matrix multiplication
 * 
 * C = A @ B
 *
 * @param out: the output tensor of shape (m, n)
 * @param a: the input tensor of shape (m, k)
 * @param b: the input tensor of shape (k, n)
 */
void NN_matmul(Tensor *out, Tensor *a, Tensor *b);

/**
 * Matrix multiplication with transposed B
 * 
 * C = A @ B
 *
 * @param out: the output tensor of shape (m, n)
 * @param a: the input tensor of shape (m, k)
 * @param b: the input tensor of shape (n, k)
 */
void NN_matmul_t(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_MATMUL_H