#ifndef __NN__NEG_H
#define __NN__NEG_H

#include <stddef.h>
#include <stdint.h>


void NN__neg_f32(size_t n,
    float *y, size_t incy,
    float *x, size_t incx
    );


#endif // __NN__NEG_H
