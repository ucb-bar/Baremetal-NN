#ifndef __NN__SOFTMAX_H
#define __NN__SOFTMAX_H

#include <stddef.h>
#include <math.h>


void NN__softmax_f32(size_t n,
    float *y, size_t incy,
    const float *x, size_t incx
    );


#endif // __NN__SOFTMAX_H
