#ifndef __NN__DOT_H
#define __NN__DOT_H

#include <stddef.h>

#include "nn_float16.h"


void NN__dot_f16(size_t n,
    float16_t *result,
    const float16_t *x, size_t incx,
    const float16_t *y, size_t incy
    );

void NN__dot_f32(size_t n,
    float *result,
    const float *x, size_t incx,
    const float *y, size_t incy
    );

#endif // __NN__DOT_H
