#ifndef __NN_SUM_H
#define __NN_SUM_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "float16.h"


void NN_sum_u8_to_i32(size_t n,
    int32_t *r,
    const uint8_t *x, size_t incx
    );

void NN_sum_i8_to_i32(size_t n,
    int32_t *r,
    const int8_t *x, size_t incx
    );

void NN_sum_i16_to_i32(size_t n,
    int32_t *r,
    const int16_t *x, size_t incx
    );

void NN_sum_i32(size_t n,
    int32_t *r,
    const int32_t *x, size_t incx
    );

void NN_sum_f16(size_t n,
    float16_t *r,
    const float16_t *x, size_t incx
    );

void NN_sum_f32(size_t n,
    float *r,
    const float *x, size_t incx);


#endif // __NN_SUM_H
