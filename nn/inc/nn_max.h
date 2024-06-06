#ifndef __NN_MAX_H
#define __NN_MAX_H

#include <assert.h>
#include <float.h>

#include "nn_tensor.h"


/**
 * Returns the maximum value of all elements in the input tensor.
 * 
 * @param t: input tensor
 */
float NN_max(Tensor *t);

float NN_max_F32(Tensor *t);

float NN_max_F32_RVV(Tensor *t);


#endif // __NN_MAX_H
