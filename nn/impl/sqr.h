#ifndef __NN__SQR_H
#define __NN__SQR_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>


void NN__sqr_f32(size_t n,
    float *y, size_t incy,
    float *x, size_t incx
    );


#endif // __NN__SQR_H
