#ifndef __NN__MINIMUM_H
#define __NN__MINIMUM_H

#include <stddef.h>
#include <stdint.h>


void NN__minimum_f32(size_t n,
    float *z, size_t incz,
    float *x, size_t incx,
    float *y, size_t incy
    );


#endif // __NN__MINIMUM_H
