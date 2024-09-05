#ifndef __NN_FILL_H
#define __NN_FILL_H

#include <assert.h>
#include <math.h>

#include "tensor.h"
#include "impl/fill.h"

/**
 * Fills the tensor with the specified value.
 * 
 * @param tensor: the input tensor
 * @param value: scalar value
 */
void NN_fill(Tensor *tensor, float value);


#endif // __NN_FILL_H
