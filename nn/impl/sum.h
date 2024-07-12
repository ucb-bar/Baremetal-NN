#ifndef __NN__SUM_H
#define __NN__SUM_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>


void NN__sum_u8_to_i32(size_t n,
    int32_t *result,
    uint8_t *x, size_t incx
    );

void NN__sum_i16_to_i32(size_t n,
    int32_t *result,
    int16_t *x, size_t incx
    );

void NN__sum_i32(size_t n,
    int32_t *result,
    int32_t *x, size_t incx
    );

void NN__sum_f32(size_t n,
    float *result,
    float *x, size_t incx);


#endif // __NN__SUM_H
