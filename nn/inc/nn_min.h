#ifndef __NN_MIN_H
#define __NN_MIN_H

#include <assert.h>
#include <float.h>

#include "nn_tensor.h"


/**
 * Returns the minimum value of all elements in the input tensor.
 * 
 * @param tensor: the input tensor
 */
float NN_min(Tensor *tensor);

float NN_min_F32(Tensor *tensor);

float NN_min_F32_RVV(Tensor *tensor);


#endif // __NN_MIN_H
