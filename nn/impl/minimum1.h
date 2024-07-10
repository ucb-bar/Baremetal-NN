#ifndef __NN__MINIMUM1_H
#define __NN__MINIMUM1_H

#include <stddef.h>
#include <stdint.h>


void NN__minimum1_f32(size_t n,
    float *y, size_t incy,
    float *x, size_t incx,
    float scalar
    );


#endif // __NN__MINIMUM1_H
