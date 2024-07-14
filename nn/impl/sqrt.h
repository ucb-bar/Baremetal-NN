#ifndef __NN__SQRT_H
#define __NN__SQRT_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>


void NN__sqrt_f32(size_t n,
    float *y, size_t incy,
    const float *x, size_t incx
    );


#endif // __NN__SQRT_H
