#ifndef __NN_NORM_H
#define __NN_NORM_H

#include <stddef.h>
#include <math.h>

#include "dot.h"


void NN_norm_f32(size_t n,
    float *result,
    const float *x, size_t incx
    );

void NN_norm_inv_f32(size_t n,
    float *result,
    const float *x, size_t incx
    );


#endif // __NN_NORM_H
