#ifndef __NN_TENSOR_CREATION
#define __NN_TENSOR_CREATION

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "nn_tensor.h"
#include "nn_fill.h"


/**
 * Initialize a given tensor
 * 
 * The memory is initialized in C order, i.e., the last dimension is contiguous.
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @param data: pointer to data, if NULL, the data will be allocated
 */
void NN_init_tensor(Tensor *tensor, const size_t ndim, const size_t *shape, DataType dtype, void *data);

/**
 * Create a new tensor
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @param data: pointer to data, if NULL, the data will be allocated
 * @return Tensor
*/
Tensor *NN_tensor(size_t ndim, const size_t *shape, DataType dtype, void *data);

/**
 * Returns a tensor filled with the scalar value 0.
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @return Tensor
 */
Tensor *NN_zeros(size_t ndim, const size_t *shape, DataType dtype);

/**
 * Returns a tensor filled with the scalar value 1.
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @return Tensor
 */
Tensor *NN_ones(size_t ndim, const size_t *shape, DataType dtype);

/**
 * Returns a tensor filled with random numbers from a uniform distribution.
 * 
 * The range of the random number is dependent on the data type:
 * - For Float32, the range is [0, 1]
 * - For Int8, the range is [0, 255]
 * - For Int32, the range is [0, RAND_MAX]
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @return Tensor
 */
Tensor *NN_rand(size_t ndim, const size_t *shape, DataType dtype);


#endif // __NN_TENSOR_CREATION