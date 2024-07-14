#ifndef __NN_INTERPOLATE_H
#define __NN_INTERPOLATE_H

#include <assert.h>
#include <string.h>

#include "nn_tensor.h"


/**
 * Down/up samples the input.
 * 
 * Tensor interpolated to either the given size or the given scale_factor
 * The algorithm used for interpolation is determined by mode.
 * 
 * @param out: the output tensor
 * @param input: the input tensor
 */
void NN_interpolate(Tensor *out, const Tensor *in, const float *scale_factor/*, const char* mode*/);


#endif // __NN_INTERPOLATE_H
