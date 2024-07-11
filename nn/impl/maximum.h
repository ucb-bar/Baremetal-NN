#ifndef __NN__MAXIMUM_H
#define __NN__MAXIMUM_H

#include <stddef.h>
#include <stdint.h>


void NN__maximum_f32(size_t n,
    float *z, size_t incz,
    float *x, size_t incx,
    float *y, size_t incy
    );


#endif // __NN__MAXIMUM_H
