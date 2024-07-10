#ifndef __NN__SGN_H
#define __NN__SGN_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>


void NN__sgn_f32(size_t n,
    float *y, size_t incy,
    float *x, size_t incx
    );


#endif // __NN__SGN_H
