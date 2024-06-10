#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>

#include "nn.h"

#define N_DIMS 2

int main() {
  Tensor *a = NN_rand(2, (size_t[]){4, 4}, DTYPE_F32);

  Tensor *ones = NN_ones(2, (size_t[]){4, 4}, DTYPE_F32);

  NN_sub(a, a, ones);

  NN_abs_F32_AVX(a, a);

  NN_printf(a);
  
}