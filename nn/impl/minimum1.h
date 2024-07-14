#ifndef __NN__MINIMUM1_H
#define __NN__MINIMUM1_H

#include <stddef.h>
#include <stdint.h>

#include "nn_float16.h"


void NN__minimum1_f32(size_t n,
    float *y, size_t incy,
    const float *x, size_t incx,
    float scalar
    );


#endif // __NN__MINIMUM1_H
