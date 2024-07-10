#ifndef __NN_SUM_H
#define __NN_SUM_H

#include <assert.h>

#include "nn_tensor.h"
#include "sum.h"


/**
 * Returns the sum of all elements in the input tensor.
 * 
 * @param out: the output scalar tensor
 * @param tensor: the input tensor
 */
void NN_sum(Tensor *out, Tensor *tensor);


#endif // __NN_SUM_H
