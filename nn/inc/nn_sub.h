#ifndef __NN_SUB_H
#define __NN_SUB_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"
#include "nn_print.h"
#include "kernel/sub.h"


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

void NN_sub_1D(Tensor *out, Tensor *a, Tensor *b);

void NN_sub_2D(Tensor *out, Tensor *a, Tensor *b);

void NN_sub_3D(Tensor *out, Tensor *a, Tensor *b);

void NN_sub_4D(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_SUB_H
