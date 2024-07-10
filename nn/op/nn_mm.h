#ifndef __NN_MATMUL_H
#define __NN_MATMUL_H

#include <assert.h>

#include "nn_tensor.h"
#include "dot.h"


/**
 * Performs a matrix multiplication.
 * 
 * If input is a (m×k) tensor, mat2 is a (k×n) tensor, out will be a (m×n) tensor.
 * 
 * C = A @ B
 *
 * @param out: the output tensor of shape (m, n)
 * @param a: the input tensor of shape (m, k)
 * @param b: the input tensor of shape (k, n)
 */
void NN_mm(Tensor *out, Tensor *a, Tensor *b);

/**
 * Performs a matrix multiplication with transposed B.
 * 
 * If input is a (m×k) tensor, mat2 is a (n×k) tensor, out will be a (m×n) tensor.
 * 
 * C = A @ B.T
 *
 * @param out: the output tensor of shape (m, n)
 * @param a: the input tensor of shape (m, k)
 * @param b: the input tensor of shape (n, k)
 */
void NN_mm_t(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_MATMUL_H
