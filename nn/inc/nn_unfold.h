#ifndef __NN_UNFOLD_H
#define __NN_UNFOLD_H

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
void NN_unfold(Tensor *data_col, Tensor *data_im, 
            const size_t *kernel_size,
            const size_t *stride, const size_t *padding, const size_t *dilation);


#endif // __NN_UNFOLD_H
