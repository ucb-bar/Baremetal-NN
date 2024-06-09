#ifndef __NN_MATMUL_H
#define __NN_MATMUL_H

#include <assert.h>

#include "nn_tensor.h"


/**
 * Matrix multiplication
 * 
 * C = A @ B
 *
 * @param c: Tensor result of shape (m, n)
 * @param a: Tensor input operand of shape (m, k)
 * @param b: Tensor input operand of shape (k, n), requires to be INT32
 */
void NN_matmul(Tensor *out, Tensor *a, Tensor *b);

void NN_matmul_F32(Tensor *out, Tensor *a, Tensor *b);

void NN_matmulT_F32(Tensor *out, Tensor *a, Tensor *b);

void NN_matmul_I8_I8_I32(Tensor *out, Tensor *a, Tensor *b);

void NN_matmul_I32(Tensor *out, Tensor *a, Tensor *b);


void NN_matmul_F32_RVV(Tensor *out, Tensor *a, Tensor *b);

void NN_matmulT_F32_RVV(Tensor *out, Tensor *a, Tensor *b);

void NN_matmul_I8_I8_I32_EAGLEX(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_MATMUL_H
