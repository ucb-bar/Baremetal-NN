#ifndef __NN__SUM_H
#define __NN__SUM_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "nn_float16.h"


void NN__sum_u8_to_i32(size_t n,
    int32_t *result,
    const uint8_t *x, size_t incx
    );

void NN__sum_i8_to_i32(size_t n,
    int32_t *result,
    const int8_t *x, size_t incx
    );

void NN__sum_i16_to_i32(size_t n,
    int32_t *result,
    const int16_t *x, size_t incx
    );

void NN__sum_i32(size_t n,
    int32_t *result,
    const int32_t *x, size_t incx
    );

void NN__sum_f16(size_t n,
    float16_t *result,
    const float16_t *x, size_t incx
    );

void NN__sum_f32(size_t n,
    float *result,
    const float *x, size_t incx);


#endif // __NN__SUM_H
