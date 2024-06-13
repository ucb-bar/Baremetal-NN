#ifndef __NN_FILL_H
#define __NN_FILL_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"
#include "kernel/fill.h"

/**
 * Fills the tensor with the specified value.
 * 
 * @param tensor: the input tensor
 * @param value: scalar value
 */
void NN_fill_F32(Tensor *tensor, float value);

void NN_fill_I32(Tensor *tensor, int32_t value);

void NN_fill_I8(Tensor *tensor, int8_t value);

Tensor *NN_zeros(size_t ndim, const size_t *shape, DataType dtype);

Tensor *NN_ones(size_t ndim, const size_t *shape, DataType dtype);

Tensor *NN_rand(size_t ndim, const size_t *shape, DataType dtype);



#endif // __NN_FILL_H
