#ifndef __NN_Linear_H
#define __NN_Linear_H

#include <assert.h>

#include "nn_tensor.h"
#include "nn_add.h"
#include "nn_transpose.h"
#include "nn_matmul.h"



/**
 * Applies a linear transformation to the incoming data
 * 
 * y = x @ w.T + b
 * 
 * @param y Output tensor of shape (1, out_features)
 * @param x Input tensor of shape (1, in_features)
 * @param w Weight tensor of shape (out_features, in_features)
 * @param b Bias tensor of shape (1, out_features)
 */
void NN_Linear_F32(Tensor *y, Tensor *x, Tensor *w, Tensor *b);

void NN_Linear_F32_RVV(Tensor *y, Tensor *x, Tensor *w, Tensor *b);

#endif // __NN_Linear_H
