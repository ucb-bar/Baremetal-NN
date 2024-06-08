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

void NN_add_F32(Tensor *out, Tensor *a, Tensor *b);

void NN_add_INT(Tensor *out, Tensor *a, Tensor *b);


void NN_add_F32_RVV(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_ADD_H
