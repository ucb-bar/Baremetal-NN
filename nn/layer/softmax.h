#ifndef __NN_SOFTMAX_H
#define __NN_SOFTMAX_H

#include <assert.h>

#include "tensor.h"
#include "impl/softmax.h"


/**
 * Returns the sum of all elements in the input tensor.
 * 
 * @param out: the output scalar tensor
 * @param tensor: the input tensor
 * @param dim: the dimension to reduce
 */
void NN_softmax(Tensor *out, const Tensor *tensor, int dim);


#endif // __NN_SOFTMAX_H
