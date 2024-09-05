#ifndef __NN_ACC1_H
#define __NN_ACC1_H

#include <stddef.h>

#include "float16.h"


void NN_acc1_i8(size_t n,
    int8_t *y, size_t incy,
    int8_t s
    );

void NN_acc1_i16(size_t n,
    int16_t *y, size_t incy,
    int16_t s
    );

void NN_acc1_i32(size_t n,
    int32_t *y, size_t incy,
    int32_t s
    );

void NN_acc1_f16(size_t n,
    float16_t *y, size_t incy,
    float16_t s
    );

void NN_acc1_f32(size_t n,
    float *y, size_t incy,
    float s
    );


#endif // __NN_ADD1_H
