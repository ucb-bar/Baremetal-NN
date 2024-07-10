#ifndef __NN__MAX_H
#define __NN__MAX_H

#include <stddef.h>
#include <stdint.h>
#include <float.h>


void NN__max_f32(size_t n,
    float *result,
    float *x, size_t incx
    );


#endif // __NN__MAX_H
