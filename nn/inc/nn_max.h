#ifndef __NN_MAX_H
#define __NN_MAX_H

#include <assert.h>
#include <float.h>

#include "nn_tensor.h"
#include "ops/max.h"


/**
 * Returns the maximum value of all elements in the input tensor.
 * 
 * @param tensor: the input tensor
 */
float NN_max(Tensor *tensor);


#endif // __NN_MAX_H
