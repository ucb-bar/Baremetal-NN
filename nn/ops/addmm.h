#ifndef __NN_ADDMM_H
#define __NN_ADDMM_H

#include <stddef.h>
#include <stdint.h>

#include "float16.h"
#include "ops/dot.h"


void NN_addmm_t_f32(size_t m, size_t n, size_t k, float *y, const float *x1, const float *x2, const float *x3);


#endif // __NN_ADDMM_H
