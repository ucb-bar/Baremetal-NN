#ifndef __NN__ACC_H
#define __NN__ACC_H

#include <stddef.h>
#include <stdint.h>


void NN__acc_i8(size_t n,
    int8_t *y, size_t incy,
    const int8_t *x, size_t incx
    );

void NN__acc_f32(size_t n,
    float *y, size_t incy,
    const float *x, size_t incx
    );


#endif // __NN__ACC_H
