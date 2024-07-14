#ifndef __NN__NORM_H
#define __NN__NORM_H

#include <stddef.h>
#include <math.h>

#include "dot.h"


void NN__norm_f32(size_t n,
    float *result,
    const float *x, size_t incx
    );

void NN__norm_inv_f32(size_t n,
    float *result,
    const float *x, size_t incx
    );


#endif // __NN__NORM_H
