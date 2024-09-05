#ifndef __NN_MIN_H
#define __NN_MIN_H

#include <stddef.h>
#include <stdint.h>
#include <float.h>

#include "float16.h"


void NN_min_i8(size_t n,
    int8_t *r,
    const int8_t *x, size_t incx
    );

void NN_min_i16(size_t n,
    int16_t *r,
    const int16_t *x, size_t incx
    );

void NN_min_i32(size_t n,
    int32_t *r,
    const int32_t *x, size_t incx
    );

void NN_min_f16(size_t n,
    float16_t *r,
    const float16_t *x, size_t incx
    );

void NN_min_f32(size_t n,
    float *r,
    const float *x, size_t incx
    );


#endif // __NN_MIN_H
