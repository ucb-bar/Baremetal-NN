
#include "nn_matrixnorm.h"

#ifdef RVV
  #include <riscv_vector.h>
#endif

void NN_matrixNorm(Tensor *scalar, Tensor *x) {
  assert(x->ndim == 2);
  assert(NN_isScalar(scalar));
  assert(scalar->dtype == x->dtype);

  switch (x->dtype) {
    case DTYPE_F32:
      NN_matrixNorm_F32(scalar, x);
      return;

    default:
      break;
  }
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = ||%s||\n", 
    NN_getDataTypeName(scalar->dtype), NN_getDataTypeName(x->dtype)
  );
}

void NN_matrixNorm_F32(Tensor *scalar, Tensor *x) {
  float sum = 0;
  #ifdef RVV
    float *ptr = x->data;

    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t vec_accumulate = __riscv_vfmv_v_f_f32m1(0, vlmax);

    size_t n = x->shape[0] * x->shape[1];
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr, vl);
      vec_accumulate = __riscv_vfmacc_vv_f32m1(vec_accumulate, vec_a, vec_a, vl);
      ptr += vl;
      n -= vl;
    }
    vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_accumulate, vec_zero, vlmax);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
  #else
    for (size_t i = 0; i < x->shape[0]; i += 1) {
      for (size_t j = 0; j < x->shape[1]; j += 1) {
        sum += pow(((float *)x->data)[i * x->shape[1] + j], 2);
      }
    }
  #endif

  ((float *)scalar->data)[0] = sqrt(sum);
  return;
}