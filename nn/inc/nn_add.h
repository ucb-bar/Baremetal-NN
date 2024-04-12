#ifndef __NN_ADD_H
#define __NN_ADD_H

#include <assert.h>

#include "nn_types.h"

/**
 * Element-wise addition
 * 
 * C = A + B
 *
 * @param out: output tensor of shape (m, n)
 * @param a: input tensor of shape (m, n)
 * @param b: input tensor of shape (m, n)
 */
void NN_add(Tensor *out, Tensor *a, Tensor *b);

void NN_add_I8(Tensor *out, Tensor *a, Tensor *b);

void NN_add_I8_I8_I32(Tensor *out, Tensor *a, Tensor *b);

void NN_add_I32_I8_I32(Tensor *out, Tensor *a, Tensor *b);

void NN_add_I32(Tensor *out, Tensor *a, Tensor *b);

void NN_add_F32(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_ADD_H
