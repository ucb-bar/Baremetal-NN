#ifndef __NN_RMS_NORM_H
#define __NN_RMS_NORM_H

#include <assert.h>
#include <math.h>

#include "tensor.h"
#include "ops/rms_norm.h"


/**
 * Computes the root mean square normalization of a tensor.
 * 
 * @param x: the input tensor of shape (m, n)
 * @param y: the output tensor of shape (m, n)
 */
void NN_rms_norm(Tensor *y, const Tensor *x, const Tensor *w, float eps);


#endif // __NN_RMS_NORM_H
