#ifndef __NN_SUM_H
#define __NN_SUM_H

#include <assert.h>

#include "nn_tensor.h"
#include "ops/sum.h"


/**
 * Returns the sum of all elements in the input tensor.
 * 
 * @param tensor: the input tensor
 */
float NN_sum_F32(Tensor *tensor);


#endif // __NN_SUM_H
