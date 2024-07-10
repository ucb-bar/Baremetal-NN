#ifndef __NN_MAXIMUM_H
#define __NN_MAXIMUM_H

#include <assert.h>
#ifdef RVV
  #include <riscv_vector.h>
#endif

#include "nn_tensor.h"
#include "maximum.h"


/**
 * Computes the element-wise maximum of two tensors.
 * 
 * @param out: the output tensor
 * @param a: the input tensor
 * @param b: the input tensor
 */
void NN_maximum(Tensor *out, Tensor *a, Tensor *b);


#endif // __NN_MAXIMUM_H
