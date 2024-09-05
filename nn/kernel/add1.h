#ifndef __NN_ADD1_H
#define __NN_ADD1_H

#include <stddef.h>

#include "float16.h"

void NN_add1_i8(size_t n,
    int8_t *z, size_t incz,
    const int8_t *x, size_t incx,
    int8_t scalar
    );

void NN_add1_i16(size_t n,
    int16_t *z, size_t incz,
    const int16_t *x, size_t incx,
    int16_t scalar
    );

void NN_add1_i32(size_t n,
    int32_t *z, size_t incz,
    const int32_t *x, size_t incx,
    int32_t scalar
    );

void NN_add1_f16(size_t n,
    float16_t *z, size_t incz,
    const float16_t *x, size_t incx,
    float16_t scalar
    );

void NN_add1_f32(size_t n,
    float *z, size_t incz,
    const float *x, size_t incx,
    float scalar
    );


#endif // __NN_ADD1_H
