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

static inline void NN__abs_i8(size_t n, int8_t *y, int8_t *x) {
  #if defined(RVV)
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m1(n);
      vint8m1_t vec_x = __riscv_vle8_v_i8m1(x, vl);
      vint8m1_t vec_neg_x = __riscv_vneg_v_i8m1(vec_x, vl);
      vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(vec_x, 0, vl);
      vint8m1_t vec_abs_x = __riscv_vmerge_vvm_i8m1(vec_x, vec_neg_x, mask, vl);
      __riscv_vse8_v_i8m1(y, vec_abs_x, vl);
      x += vl;
      y += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      y[i] = x[i] < 0 ? -x[i] : x[i];
    }
  #endif
}

static inline void NN__abs_i16(size_t n, int16_t *y, int16_t *x) {
  #if defined(RVV)
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vint16m1_t vec_x = __riscv_vle16_v_i16m1(x, vl);
      vint16m1_t vec_neg_x = __riscv_vneg_v_i16m1(vec_x, vl);
      vbool16_t mask = __riscv_vmslt_vx_i16m1_b16(vec_x, 0, vl);
      vint16m1_t vec_abs_x = __riscv_vmerge_vvm_i16m1(vec_x, vec_neg_x, mask, vl);
      __riscv_vse16_v_i16m1(y, vec_abs_x, vl);
      x += vl;
      y += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      y[i] = x[i] < 0 ? -x[i] : x[i];
    }
  #endif
}

static inline void NN__abs_i32(size_t n, int32_t *y, int32_t *x) {
  #if defined(RVV)
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vint32m1_t vec_x = __riscv_vle32_v_i32m1(x, vl);
      vint32m1_t vec_neg_x = __riscv_vneg_v_i32m1(vec_x, vl);
      vbool32_t mask = __riscv_vmslt_vx_i32m1_b32(vec_x, 0, vl);
      vint32m1_t vec_abs_x = __riscv_vmerge_vvm_i32m1(vec_x, vec_neg_x, mask, vl);
      __riscv_vse32_v_i32m1(y, vec_abs_x, vl);
      x += vl;
      y += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      y[i] = x[i] < 0 ? -x[i] : x[i];
    }
  #endif
}

static inline void NN__abs_f16(size_t n, float16_t *y, float16_t *x) {
  for (size_t i = 0; i < n; i += 1) {
    y[i] = NN_floatToHalf(fabsf(NN_halfToFloat(x[i])));
  }
}

static inline void NN__abs_f32(size_t n, float *y, float *x) {
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
