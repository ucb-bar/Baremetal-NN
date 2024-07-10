#include <riscv_vector.h>
#include "dot.h"

#ifdef RVV


// void NN__dot_f16(size_t n, float16_t *result, float16_t *x, size_t incx, float16_t *y, size_t incy) {
//   size_t vlmax = __riscv_vsetvlmax_e16m1();

//   vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
//   vfloat16m1_t vec_r = __riscv_vfmv_v_f_f16m1(0, vlmax);

//   while (n > 0) {
//     size_t vl = __riscv_vsetvl_e16m1(n);
//     vfloat16m1_t vec_x = __riscv_vlse16_v_f16m1(x, incx, vl);
//     vfloat16m1_t vec_y = __riscv_vlse16_v_f16m1(y, incy, vl);
//     vec_r = __riscv_vfmacc_vv_f16m1(vec_r, vec_x, vec_y, vl);
      
//     x += vl;
//     y += vl;
//     n -= vl;
//   }
//   vec_r = __riscv_vfredusum_vs_f16m1_f16m1(vec_r, vec_zero, vlmax);
//   *result = __riscv_vfmv_f_s_f16m1_f16(vec_r);
// }

void NN__dot_f32(size_t n, float *result, float *x, size_t incx, float *y, size_t incy) {
  size_t vlmax = __riscv_vsetvlmax_e32m1();

  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
  vfloat32m1_t vec_r = __riscv_vfmv_v_f_f32m1(0, vlmax);

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, incx, vl);
    vfloat32m1_t vec_y = __riscv_vlse32_v_f32m1(y, incy, vl);
    vec_r = __riscv_vfmacc_vv_f32m1(vec_r, vec_x, vec_y, vl);
      
    x += vl;
    y += vl;
    n -= vl;
  }
  vec_r = __riscv_vfredusum_vs_f32m1_f32m1(vec_r, vec_zero, vlmax);
  *result = __riscv_vfmv_f_s_f32m1_f32(vec_r);
}


#endif
