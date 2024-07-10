#ifndef __NN__MAXIMUM1_H
#define __NN__MAXIMUM1_H

#include <stddef.h>
#include <stdint.h>


void NN__maximum1_f32(size_t n,
    float *y, size_t incy,
    float *x, size_t incx,
    float scalar
    );


#endif // __NN__MAXIMUM1_H
