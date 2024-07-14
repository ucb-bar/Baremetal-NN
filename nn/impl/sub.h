#ifndef __NN__SUB_H
#define __NN__SUB_H

#include <stddef.h>
#include <stdint.h>

#include "nn_float16.h"


void NN__sub_u8(size_t n,
    uint8_t *z, size_t incz,
    const uint8_t *x, size_t incx,
    const uint8_t *y, size_t incy
    );

void NN__sub_i8(size_t n,
    int8_t *z, size_t incz,
    const int8_t *x, size_t incx,
    const int8_t *y, size_t incy
    );

void NN__sub_i16(size_t n,
    int16_t *z, size_t incz,
    const int16_t *x, size_t incx,
    const int16_t *y, size_t incy
    );

void NN__sub_i32(size_t n,
    int32_t *z, size_t incz,
    const int32_t *x, size_t incx,
    const int32_t *y, size_t incy
    );

void NN__sub_f16(size_t n,
    float16_t *z, size_t incz,
    const float16_t *x, size_t incx,
    const float16_t *y, size_t incy
    );

void NN__sub_f32(size_t n,
    float *z, size_t incz,
    const float *x, size_t incx,
    const float *y, size_t incy
    );


#endif // __NN__SUB_H
