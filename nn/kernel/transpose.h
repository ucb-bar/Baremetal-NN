#ifndef __NN_TRANSPOSE_H
#define __NN_TRANSPOSE_H

#include <stddef.h>
#include <stdint.h>

#include "float16.h"


void NN_transpose_i8(size_t m, size_t n,
    int8_t *y, 
    const int8_t *x
    );

void NN_transpose_i16(size_t m, size_t n,
    int16_t *y, 
    const int16_t *x
    );

void NN_transpose_i32(size_t m, size_t n,
    int32_t *y, 
    const int32_t *x
    );

void NN_transpose_f16(size_t m, size_t n,
    float16_t *y, 
    const float16_t *x
    );

void NN_transpose_f32(size_t m, size_t n,
    float *y, 
    const float *x
    );


#endif // __NN_TRANSPOSE_H
