#include "log.h"


__attribute__((weak)) void NN__log_f32(size_t n, float *y, size_t incy, float *x, size_t incx) {
  for (size_t i = 0; i < n; i += 1) {
    y[i * incy] = logf(x[i * incx]);
  }
}

