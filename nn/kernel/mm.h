#ifndef __NN_MM_H
#define __NN_MM_H

#include <stddef.h>
#include <stdint.h>

#include "float16.h"


void NN_mm_t_f32(size_t m, size_t n, size_t k, float *y, const float *x1, const float *x2);


#endif // __NN_MM_H
