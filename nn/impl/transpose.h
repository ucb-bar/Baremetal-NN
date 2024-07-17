#ifndef __NN__TRANSPOSE_H
#define __NN__TRANSPOSE_H

#include <stddef.h>
#include <stdint.h>

#include "nn_float16.h"


void NN__transpose_i8(size_t m, size_t n,
    int8_t *y, 
    const int8_t *x
    );

void NN__transpose_i16(size_t m, size_t n,
    int16_t *y, 
    const int16_t *x
    );

void NN__transpose_i32(size_t m, size_t n,
    int32_t *y, 
    const int32_t *x
    );

void NN__transpose_f16(size_t m, size_t n,
    float16_t *y, 
    const float16_t *x
    );

void NN__transpose_f32(size_t m, size_t n,
    float *y, 
    const float *x
    );


#endif // __NN__TRANSPOSE_H
