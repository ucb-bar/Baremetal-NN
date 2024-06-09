#ifndef __NN_NEG_H
#define __NN_NEG_H

#include <assert.h>

#include "nn_tensor.h"


/**
 * Returns a tensor with the negative of the elements of input.
 * 
 * out = -1 x input
 * 
 * @param out: the output tensor
 * @param input: the input tensor
 */
void NN_neg(Tensor *out, Tensor *input);

void NN_neg_F32(Tensor *out, Tensor *input);


void NN_neg_F32_RVV(Tensor *out, Tensor *input);


#endif // __NN_NEG_H
