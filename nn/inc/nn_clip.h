#ifndef __NN_CLIP_H
#define __NN_CLIP_H

#include <assert.h>

#include "nn_types.h"


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
void NN_clip(Tensor *t, float min, float max);


#endif // __NN_CLIP_H
