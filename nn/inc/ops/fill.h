#ifndef __NN__FILL_H
#define __NN__FILL_H

#include <stddef.h>
#include <stdint.h>

#ifdef RVV
  #include <riscv_vector.h>
#endif

#include "nn_float16.h"

inline static void NN__fill_I8(size_t n, int8_t *x, int8_t v) {
  for (size_t i = 0; i < n; i += 1) {
    x[i] = v;
  }
}

inline static void NN__fill_I16(size_t n, int16_t *x, int16_t v) {
  for (size_t i = 0; i < n; i += 1) {
    x[i] = v;
  }
}

inline static void NN__fill_I32(size_t n, int32_t *x, int32_t v) {
  for (size_t i = 0; i < n; i += 1) {
    x[i] = v;
  }
}

inline static void NN__fill_F16(size_t n, float16_t *x, float16_t v) {
  for (size_t i = 0; i < n; i += 1) {
    x[i] = v;
  }
}

// inline static void NN__fill_BF16(size_t n, bfloat16_t * x, const bfloat16_t v) {
//   for (size_t i = 0; i < n; i += 1) {
//     x[i] = v;
//   }
// }

inline static void NN__fill_F32(size_t n, float *x, float v) {
  #ifdef RVV
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vfmv_v_f_f32m1(v, vl);
      __riscv_vse32_v_f32m1(x, vec_x, vl);
      x += vl;
      n -= vl;
    }
  #else
    for (size_t i = 0; i < n; i += 1) {
      x[i] = v;
    }
  #endif
}


#endif // __NN__FILL_H
