#ifndef __NN_DOT_H
#define __NN_DOT_H

#include <stddef.h>

#include "float16.h"


void NN_dot_i8_to_i32(size_t n,
    int32_t *r,
    const int8_t *x1, size_t incx1,
    const int8_t *x2, size_t incx2
    );

void NN_dot_i16_to_i32(size_t n,
    int32_t *r,
    const int16_t *x1, size_t incx1,
    const int16_t *x2, size_t incx2
    );

void NN_dot_i32(size_t n,
    int32_t *r,
    const int32_t *x1, size_t incx1,
    const int32_t *x2, size_t incx2
    );

void NN_dot_f16(size_t n,
    float16_t *r,
    const float16_t *x1, size_t incx1,
    const float16_t *x2, size_t incx2
    );

void NN_dot_f32(size_t n,
    float *r,
    const float *x1, size_t incx1,
    const float *x2, size_t incx2
    );

#endif // __NN_DOT_H
