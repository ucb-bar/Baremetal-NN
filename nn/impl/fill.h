#ifndef __NN__FILL_H
#define __NN__FILL_H

#include <stddef.h>
#include <stdint.h>

#include "nn_float16.h"


void NN__fill_u8(size_t n,
    uint8_t *x, size_t incx,
    uint8_t scalar
    );

void NN__fill_i8(size_t n,
    int8_t *x, size_t incx,
    int8_t scalar
    );

void NN__fill_i16(size_t n,
    int16_t *x, size_t incx,
    int16_t scalar
    );

void NN__fill_i32(size_t n,
    int32_t *x, size_t incx,
    int32_t scalar
    );

void NN__fill_f16(size_t n,
    float16_t *x, size_t incx,
    float16_t scalar
    );

void NN__fill_f32(size_t n,
    float *x, size_t incx,
    float scalar
    );


#endif // __NN__FILL_H
