#ifndef __NN_SUB_H
#define __NN_SUB_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"
#include "nn_print.h"
#include "impl/sub.h"


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
void NN_sub(Tensor *out, const Tensor *a, const Tensor *b);

void NN_sub_1d(Tensor *out, const Tensor *a, const Tensor *b);

void NN_sub_2d(Tensor *out, const Tensor *a, const Tensor *b);

void NN_sub_3d(Tensor *out, const Tensor *a, const Tensor *b);

void NN_sub_4d(Tensor *out, const Tensor *a, const Tensor *b);


#endif // __NN_SUB_H
