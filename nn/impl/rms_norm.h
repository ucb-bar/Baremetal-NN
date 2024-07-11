#ifndef __NN__RMS_NORM_H
#define __NN__RMS_NORM_H

#include <stddef.h>
#include <math.h>

#include "sqr.h"
#include "sum.h"
#include "mul1.h"
#include "mul.h"

void NN__rms_norm_f32(size_t n,
    float* y, size_t incy,
    float* x, size_t incx,
    float* w, size_t incw,
    float* eps
    );


#endif // __NN__RMS_NORM_H
