#ifndef __NN_SOFTMAX_H
#define __NN_SOFTMAX_H

#include <assert.h>

#include "nn_tensor.h"
#include "ops/softmax.h"


/**
 * Returns the sum of all elements in the input tensor.
 * 
 * @param out: the output scalar tensor
 * @param tensor: the input tensor
 */
void NN_Softmax(Tensor *out, Tensor *tensor, size_t dim);


#endif // __NN_SOFTMAX_H
