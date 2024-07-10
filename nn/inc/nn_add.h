#ifndef __NN_ADD_H
#define __NN_ADD_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"
#include "nn_print.h"
#include "ops/acc.h"
#include "ops/acc1.h"
#include "ops/add.h"
#include "ops/add1.h"


/**
 * Element-wise addition
 * 
 * C = A + B
 * 
 * For broadcast, b should be smaller dimension or shape than a.
 *
 * @param out: the output tensor
 * @param a: the input tensor
 * @param b: the input tensor
 */
void NN_add(Tensor *out, Tensor *a, Tensor *b);

/**
 * Returns the element-wise addition of the input tensor with a scalar.
 * 
 * @param out: the output tensor
 * @param in: the input tensor
 * @param scalar: scalar value
 */
void NN_add1(Tensor *out, Tensor *in, float scalar);

/**
 * Element-wise addition
 * 
 * B += A
 * 
 * For broadcast, b should be smaller dimension or shape than a.
 *
 * @param b: the target tensor
 * @param a: the input tensor
 */
void NN_add_inplace(Tensor *b, Tensor *a);

/**
 * Returns the element-wise addition of the input tensor with a scalar.
 * 
 * @param b: the target tensor
 * @param scalar: scalar value
 */
void NN_add_inplace1(Tensor *b, float scalar);

void NN_add_1d(Tensor *out, Tensor *a, Tensor *b);

void NN_add_2d(Tensor *out, Tensor *a, Tensor *b);

void NN_add_3d(Tensor *out, Tensor *a, Tensor *b);

void NN_add_4d(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_ADD_H
