#ifndef __NN_MATMUL_H
#define __NN_MATMUL_H

#include <assert.h>

#include "nn_tensor.h"


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

void NN_matmul_F32(Tensor *out, Tensor *a, Tensor *b);

/**
 * Matrix multiplication with transposed B
 * 
 * C = A @ B
 *
 * @param out: the output tensor of shape (m, n)
 * @param a: the input tensor of shape (m, k)
 * @param b: the input tensor of shape (n, k)
 */
void NN_matmulT(Tensor *out, Tensor *a, Tensor *b);

void NN_matmulT_F16(Tensor *out, Tensor *a, Tensor *b);

void NN_matmulT_F32(Tensor *out, Tensor *a, Tensor *b);

void NN_matmul_I8_I8_I32(Tensor *out, Tensor *a, Tensor *b);

void NN_matmul_I32(Tensor *out, Tensor *a, Tensor *b);


void NN_matmul_F32_RVV(Tensor *out, Tensor *a, Tensor *b);

void NN_matmulT_F32_RVV(Tensor *out, Tensor *a, Tensor *b);

void NN_matmul_I8_I8_I32_EAGLEX(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_MATMUL_H
