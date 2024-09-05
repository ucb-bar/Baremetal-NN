#ifndef __NN_MIN_H
#define __NN_MIN_H

#include <assert.h>
#include <float.h>

#include "tensor.h"
#include "kernel/min.h"


/**
 * Returns the minimum value of all elements in the input tensor.
 * 
 * @param out: the output scalar tensor
 * @param tensor: the input tensor
 */
void NN_min(Tensor *scalar, const Tensor *tensor);


#endif // __NN_MIN_H
