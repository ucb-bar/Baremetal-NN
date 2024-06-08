#ifndef __NN_SUB_H
#define __NN_SUB_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"
#include "nn_print.h"


/**
 * Element-wise subtraction
 * 
 * C = A - B
 * 
 * Broadcast is only supported between tensor with same dimensions.
 *
 * @param out: the output tensor
 * @param a: the input tensor
 * @param b: the input tensor
 */
void NN_sub(Tensor *out, Tensor *a, Tensor *b);

void NN_sub_F32(Tensor *out, Tensor *a, Tensor *b);

void NN_sub_INT(Tensor *out, Tensor *a, Tensor *b);


void NN_sub_F32_RVV(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_SUB_H
