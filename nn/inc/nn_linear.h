#ifndef __NN_LINEAR_H
#define __NN_LINEAR_H

#include <assert.h>

#include "nn_types.h"
#include "nn_add.h"
#include "nn_matmul.h"


/**
 * Applies a linear transformation to the incoming data: y = x @ w.T + b
 * 
 */
void NN_linear_F32(Tensor *y, Tensor *x, Tensor *w, Tensor *b);

#endif // __NN_LINEAR_H
