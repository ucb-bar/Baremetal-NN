#include <riscv_vector.h>
#include "rms_norm.h"

#ifdef RVV


void NN__rms_norm_f32(size_t n, float* y, size_t incy, float* x, size_t incx, float* w, size_t incw, float *eps) {
  // TODO: for some reason, passing eps as float is not working, the function is getting random values
  // so we are passing it as a pointer and dereferencing it here temporarily

  // calculate sum of squares
  float ss;
  NN__sqr_f32(n, y, incy, x, incx);
  NN__sum_f32(n, &ss, y, incy);

  ss /= n;
  ss += *eps;
  ss = 1.0f / sqrtf(ss);

  // normalize and scale
  NN__mul1_f32(n, y, incy, x, incx, ss);
  NN__mul_f32(n, y, incy, y, incy, w, incw);
}

#endif
