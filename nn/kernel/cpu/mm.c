#include "kernel/mm.h"


__attribute__((weak)) void NN_mm_t_f32(size_t m, size_t n, size_t k, float *y, const float *x1, const float *x2) {
  for (size_t i = 0; i < m; i += 1) {
    for (size_t j = 0; j < n; j += 1) {
      NN_dot_f32(k, 
        (float *)y + i * n + j,
        (float *)x1 + i * k, 1,
        (float *)x2 + j * k, 1
        );
    }
  }
}
