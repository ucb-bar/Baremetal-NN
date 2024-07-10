#ifndef __NN__LOG_H
#define __NN__LOG_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>


void NN__log_f32(size_t n,
    float *y, size_t incy,
    float *x, size_t incx
    );


#endif // __NN__LOG_H
