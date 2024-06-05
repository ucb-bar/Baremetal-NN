#ifndef __NN_ADD_H
#define __NN_ADD_H

#include <assert.h>

#include "nn_tensor.h"


/**
 * Element-wise addition
 * 
 * C = A + B
 * 
 * Broadcast is only supported on input tensor B with shape (1, n)
 *
 * @param out: output tensor of shape (m, n)
 * @param a: input tensor of shape (m, n)
 * @param b: input tensor of shape (m, n) or (1, n)
 */
void NN_add(Tensor *out, Tensor *a, Tensor *b);

void NN_add_F32(Tensor *out, Tensor *a, Tensor *b);

void NN_add_INT(Tensor *out, Tensor *a, Tensor *b);


void NN_add_F32_RVV(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_ADD_H
