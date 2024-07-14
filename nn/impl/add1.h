#ifndef __NN__ADD1_H
#define __NN__ADD1_H

#include <stddef.h>


void NN__add1_f32(size_t n,
    float *z, size_t incz,
    const float *x, size_t incx,
    float scalar
    );


#endif // __NN__ADD1_H
