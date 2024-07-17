#ifndef __NN__SOFTMAX_H
#define __NN__SOFTMAX_H

#include <stddef.h>
#include <math.h>

#include "nn_float16.h"


void NN__softmax_f16(size_t n,
    float16_t *y, size_t incy,
    const float16_t *x, size_t incx
    );

void NN__softmax_f32(size_t n,
    float *y, size_t incy,
    const float *x, size_t incx
    );


#endif // __NN__SOFTMAX_H
