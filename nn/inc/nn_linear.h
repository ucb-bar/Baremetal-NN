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
void NN_Linear(Tensor *y, Tensor *x, Tensor *w, Tensor *b);
<<<<<<< HEAD

void NN_Linear_F16(Tensor *y, Tensor *x, Tensor *w, Tensor *b);

void NN_Linear_F32(Tensor *y, Tensor *x, Tensor *w, Tensor *b);


void NN_Linear_F32_RVV(Tensor *y, Tensor *x, Tensor *w, Tensor *b);
=======

>>>>>>> 264c7cf53a6ac215f54b2357d7068f11e3624c3d

#endif // __NN_Linear_H
