#ifndef __NN_MM_H
#define __NN_MM_H

#include <stddef.h>
#include <math.h>

#include "float16.h"

#include "gemmini.h"


void NN_mm_f32(size_t m, size_t n,
    float16_t *z,
    float16_t *x,
    float16_t *y
    );


#endif // __NN_MM_H
