#ifndef __NN__MINIMUM1_H
#define __NN__MINIMUM1_H

#include <stddef.h>
#include <stdint.h>

#include "nn_float16.h"


void NN__minimum1_i8(size_t n,
    int8_t *y, size_t incy,
    const int8_t *x, size_t incx,
    int8_t scalar
    );

void NN__minimum1_i16(size_t n,
    int16_t *y, size_t incy,
    const int16_t *x, size_t incx,
    int16_t scalar
    );

void NN__minimum1_i32(size_t n,
    int32_t *y, size_t incy,
    const int32_t *x, size_t incx,
    int32_t scalar
    );

void NN__minimum1_f16(size_t n,
    float16_t *y, size_t incy,
    const float16_t *x, size_t incx,
    float16_t scalar
    );

void NN__minimum1_f32(size_t n,
    float *y, size_t incy,
    const float *x, size_t incx,
    float scalar
    );


#endif // __NN__MINIMUM1_H
