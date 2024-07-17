#ifndef __NN__ACC1_H
#define __NN__ACC1_H

#include <stddef.h>

#include "nn_float16.h"


void NN__acc1_i8(size_t n,
    int8_t *result, size_t incr,
    int8_t scalar
    );

void NN__acc1_i16(size_t n,
    int16_t *result, size_t incr,
    int16_t scalar
    );

void NN__acc1_i32(size_t n,
    int32_t *result, size_t incr,
    int32_t scalar
    );

void NN__acc1_f16(size_t n,
    float16_t *result, size_t incr,
    float16_t scalar
    );

void NN__acc1_f32(size_t n,
    float *result, size_t incr,
    float scalar
    );


#endif // __NN__ADD1_H
