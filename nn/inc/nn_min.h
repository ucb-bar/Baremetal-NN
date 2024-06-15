#ifndef __NN_MIN_H
#define __NN_MIN_H

#include <assert.h>
#include <float.h>

#include "nn_tensor.h"
#include "ops/min.h"


/**
 * Returns the minimum value of all elements in the input tensor.
 * 
 * @param tensor: the input tensor
 */
float NN_min(Tensor *tensor);


#endif // __NN_MIN_H
