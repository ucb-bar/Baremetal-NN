#ifndef __NN__MAXIMUM1_H
#define __NN__MAXIMUM1_H

#include <stddef.h>
#include <stdint.h>

#include "nn_float16.h"


void NN__maximum1_f16(size_t n,
    float16_t *y, size_t incy,
    const float16_t *x, size_t incx,
    float16_t scalar
    );

void NN__maximum1_f32(size_t n,
    float *y, size_t incy,
    const float *x, size_t incx,
    float scalar
    );


#endif // __NN__MAXIMUM1_H
