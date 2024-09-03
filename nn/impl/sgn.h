#ifndef __NN_SGN_H
#define __NN_SGN_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>


void NN_sgn_f32(size_t n,
    float *y, size_t incy,
    const float *x, size_t incx
    );


#endif // __NN_SGN_H
