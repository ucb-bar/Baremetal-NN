#ifndef __NN_SQRT_H
#define __NN_SQRT_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "float16.h"


void NN_sqrt_f16(size_t n,
    float16_t *y, size_t incy,
    const float16_t *x, size_t incx
    );

void NN_sqrt_f32(size_t n,
    float *y, size_t incy,
    const float *x, size_t incx
    );


#endif // __NN_SQRT_H
