
#include "nn_abs.h"
#include <immintrin.h>

void NN_abs_F32_AVX(Tensor *out, Tensor *input) {
  assert(out->ndim == input->ndim);
  assert(input->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->size == input->size);

  float *ptr_out = out->data;
  float *ptr_in = input->data;
  
  size_t n = out->shape[0] * out->shape[1];
  size_t vl = 8;

  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));  // Mask to clear the sign bit

  while (n > 0) {
    size_t count = n < vl ? n : vl;

    // Load input values into an AVX register
    __m256 vec_in = _mm256_loadu_ps(ptr_in);
    
    // Compute the absolute values
    __m256 vec_out = _mm256_and_ps(vec_in, mask);
    
    // Store the result
    _mm256_storeu_ps(ptr_out, vec_out);
    
    ptr_in += count;
    ptr_out += count;
    n -= count;
  }
}

