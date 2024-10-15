
#include <immintrin.h>

#include "abs.h"

#ifdef AVX

// void nn_abs_f32(size_t n, float *result, float *x, size_t incx) {
//   // Mask to clear the sign bit
//   __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

//   size_t vl = 8;

//   while (n > 0) {
//     size_t count = n < vl ? n : vl;
//     // Load input values into an AVX register
//     __m256 vec_x = _mm256_loadu_ps(x);
//     // Compute the absolute values
//     __m256 vec_y = _mm256_and_ps(vec_x, mask);
//     // Store the result
//     _mm256_storeu_ps(y, vec_y);
//     x += count;
//     y += count;
//     n -= count;
//   }
// }

#endif