#ifndef __NN_CLIP_H
#define __NN_CLIP_H

#include <assert.h>

#include "tensor.h"
#include "impl/maximum1.h"
#include "impl/minimum1.h"


/**
 * Clamps all elements in input into the range [ min, max ]. 
 * Letting min_value and max_value be min and max, respectively, this returns:
 * 
 *   y_i = min(max(x_i, min_value), max_value)
 * 
 * @param out: the output tensor
 * @param a: the input tensor
 * @param min: lower-bound of the range to be clamped to
 * @param max: upper-bound of the range to be clamped to
 */
void NN_clip(Tensor *y, const Tensor *x, float min, float max);

void NN_clip_inplace(Tensor *x, float min, float max);


#endif // __NN_CLIP_H
