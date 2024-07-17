#include "mm.h"


void NN__mm_f32(size_t m, size_t n, float16_t *z, float16_t *x, float16_t *y) {
  size_t dim_I = m;
  size_t dim_J = n;
  size_t dim_K = n;

  size_t stride_A = dim_K;
  size_t stride_B = dim_J;
  size_t stride_D = dim_J;
  size_t stride_C = dim_J;

  tiled_matmul_auto(dim_I, dim_J, dim_K,
      x, y,
      NULL, z,
      stride_A, stride_B, stride_D, stride_C,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
      0, 0, 0, 0, 0, 0, WS);
  
  return;
};
