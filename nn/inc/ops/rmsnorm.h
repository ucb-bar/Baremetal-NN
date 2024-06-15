#ifndef __NN__RMSNORM_H
#define __NN__RMSNORM_H

#include <stddef.h>
#include <math.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

void NN__rmsnorm_F32(size_t size, float* o, float* x, float* weight) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

#endif // __NN__RMSNORM_H
