#ifndef __NN_FILL_H
#define __NN_FILL_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"
#include "ops/fill.h"

/**
 * Fills the tensor with the specified value.
 * 
 * @param tensor: the input tensor
 * @param value: scalar value
 */
void NN_fill(Tensor *tensor, float value);

Tensor *NN_zeros(size_t ndim, const size_t *shape, DataType dtype);

Tensor *NN_ones(size_t ndim, const size_t *shape, DataType dtype);

Tensor *NN_rand(size_t ndim, const size_t *shape, DataType dtype);



#endif // __NN_FILL_H
