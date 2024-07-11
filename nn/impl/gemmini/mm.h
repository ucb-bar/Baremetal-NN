#ifndef __NN__MM_H
#define __NN__MM_H

#include <stddef.h>
#include <math.h>

#include "nn_float16.h"

#include "gemmini.h"


void NN__mm_f32(size_t m, size_t n,
    float16_t *z,
    float16_t *x,
    float16_t *y
    );


#endif // __NN__MM_H