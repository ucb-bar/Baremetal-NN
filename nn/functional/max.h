#ifndef __NN_MAX_H
#define __NN_MAX_H

#include <assert.h>
#include <float.h>

#include "tensor.h"
#include "ops/max.h"


/**
 * Returns the maximum value of all elements in the input tensor.
 * 
 * @param out: the output scalar tensor
 * @param tensor: the input tensor
 */
void NN_max(Tensor *scalar, const Tensor *tensor);


#endif // __NN_MAX_H
