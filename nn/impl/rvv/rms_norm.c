#include <riscv_vector.h>
#include "impl/rms_norm.h"

#ifdef RVV

void NN__rms_norm_f32(size_t n, float* y, size_t incy, const float* x, size_t incx, const float* w, size_t incw, float eps) {
  // calculate sum of squares
  
  float ss = 0.0f;
  
  // this is somehow not working
  // NN__sqr_f32(n, y, incy, x, incx);
  // NN__sum_f32(n, &ss, y, incy);
  
  for (size_t i = 0; i < n; i += 1) {
    ss += x[i * incx] * x[i * incx];
  }
  ss /= n;
  ss += eps;

  // normalize and scale
  // y = (x / ss) * w
  NN__mul1_f32(n, y, incy, x, incx, 1.0f / sqrtf(ss));
  NN__mul_f32(n, y, incy, y, incy, w, incw);
}

#endif
