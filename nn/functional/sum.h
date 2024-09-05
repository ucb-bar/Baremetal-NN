#ifndef __NN_SUM_H
#define __NN_SUM_H

#include <assert.h>

#include "tensor.h"
#include "kernel/sum.h"


/**
 * Returns the sum of all elements in the input tensor.
 * 
 * @param out: the output scalar tensor
 * @param tensor: the input tensor
 */
void NN_sum(Tensor *out, const Tensor *tensor);


#endif // __NN_SUM_H
