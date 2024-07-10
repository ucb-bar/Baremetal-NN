#ifndef __NN__RMSNORM_H
#define __NN__RMSNORM_H

#include <stddef.h>
#include <math.h>


void NN__rmsnorm_f32(size_t n,
    float* y, size_t incy,
    float* x, size_t incx,
    float* w, size_t incw
    );


#endif // __NN__RMSNORM_H
