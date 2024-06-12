#ifndef __NN_ADD_H
#define __NN_ADD_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"
#include "nn_print.h"


/**
 * Element-wise addition
 * 
 * C = A + B
 * 
 * Broadcast is only supported between tensor with same dimensions.
 *
 * @param out: the output tensor
 * @param a: the input tensor
 * @param b: the input tensor
 */
void NN_add(Tensor *out, Tensor *a, Tensor *b);

void NN_add_F16(Tensor *out, Tensor *a, Tensor *b);

void NN_add_F32(Tensor *out, Tensor *a, Tensor *b);

void NN_add_INT(Tensor *out, Tensor *a, Tensor *b);

/**
 * Returns the element-wise addition of the input tensor with a scalar.
 * 
 * @param out: the output tensor
 * @param in: the input tensor
 * @param scalar: scalar value
 */
void NN_addF(Tensor *out, Tensor *in, float scalar);

void NN_addF_F32(Tensor *out, Tensor *in, float scalar);


void NN_add_F32_RVV(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_ADD_H
