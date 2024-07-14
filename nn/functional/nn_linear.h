#ifndef __NN_Linear_H
#define __NN_Linear_H

#include <assert.h>

#include "nn_tensor.h"
#include "nn_matmul.h"
#include "nn_add.h"


/**
 * Applies a linear transformation to the incoming data
 * 
 * y = x @ w.T + b
 * 
 * @param y: the output tensor of shape (1, out_features)
 * @param x: tnput tensor of shape (1, in_features)
 * @param w: weight tensor of shape (out_features, in_features)
 * @param b: bias tensor of shape (1, out_features), or NULL if no bias is applied
 */
void NN_linear(Tensor *y, const Tensor *x, const Tensor *w, const Tensor *b);

#endif // __NN_Linear_H
