#ifndef __NN_COPY_H
#define __NN_COPY_H

#include <assert.h>
#include <string.h>

#include "nn_tensor.h"


/**
 * Copies values from one tensor to another
 * 
 * If the data types of the two tensors are different, the values are casted to the destination data type
 * 
 * @param dst: destination tensor
 * @param src: source tensor
 */
void NN_copy(Tensor *dst, Tensor *src);


#endif // __NN_COPY_H
