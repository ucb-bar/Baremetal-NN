#ifndef __NN__ABS_H
#define __NN__ABS_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifdef AVX
  #include <immintrin.h>
#endif

#ifdef RVV
  #include <riscv_vector.h>
#endif

static inline void NN__abs_F32(size_t n, float *y, float *x) {
  #if defined(AVX)
    // Mask to clear the sign bit
    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    size_t vl = 8;

    while (n > 0) {
      size_t count = n < vl ? n : vl;

      // Load input values into an AVX register
      __m256 vec_x = _mm256_loadu_ps(x);
      
      // Compute the absolute values
      __m256 vec_y = _mm256_and_ps(vec_x, mask);
      
      // Store the result
      _mm256_storeu_ps(y, vec_y);
      
      x += count;
      y += count;
      n -= count;
    }
  #elif defined(RVV)
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
      vfloat32m1_t vec_y = __riscv_vfabs_v_f32m1(vec_x, vl);
      __riscv_vse32_v_f32m1(y, vec_y, vl);
      x += vl;
      y += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      y[i] = fabsf(x[i]);
    }
  #endif
}

#endif // __NN__ABS_H
