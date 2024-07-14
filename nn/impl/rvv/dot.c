#include <riscv_vector.h>
#include "dot.h"

#ifdef RVV


void NN__dot_f16(size_t n, float16_t *result, const float16_t *x, size_t incx, const float16_t *y, size_t incy) {
  size_t vlmax;
  // size_t vlmax = __riscv_vsetvlmax_e16m1();
  asm volatile("vsetvli %0, zero, e16, m1, ta, ma" : "=r"(vlmax) : "r"(n));

  // vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
  asm volatile("vmv.v.i v27, 0");

  // vfloat16m1_t vec_r = __riscv_vfmv_v_f_f16m1(0, vlmax);
  asm volatile("vmv1r.v v24, v27");

  while (n > 0) {
    size_t vl;
    // size_t vl = __riscv_vsetvl_e16m1(n);
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(n));

    // vfloat16m1_t vec_x = __riscv_vlse16_v_f16m1(x, sizeof(float16_t) * incx, vl);
    asm volatile("vlse16.v v26, (%0), %1" : : "r"(x), "r"(sizeof(float16_t) * incx));

    // vfloat16m1_t vec_y = __riscv_vlse16_v_f16m1(y, sizeof(float16_t) * incy, vl);
    asm volatile("vlse16.v v25, (%0), %1" : : "r"(y), "r"(sizeof(float16_t) * incy));
    
    // vec_r = __riscv_vfmacc_vv_f16m1(vec_r, vec_x, vec_y, vl);
    asm volatile("vfmacc.vv v24, v26, v25");
      
    x += vl;
    y += vl;
    n -= vl;
  }

  // vec_r = __riscv_vfredusum_vs_f16m1_f16m1(vec_r, vec_zero, vlmax);
  asm volatile("vsetvli %0, zero, e16, m1, ta, ma" : "=r"(vlmax) : "r"(n));
  asm volatile("vfredusum.vs v24, v24, v27");

  // *result = __riscv_vfmv_f_s_f16m1_f16(vec_r);
  float16_t r;
  asm volatile("vmv.x.s %0, v24" : "=r"(r));
  *result = r;
  
}

void NN__dot_f32(size_t n, float *result, const float *x, size_t incx, const float *y, size_t incy) {
  size_t vlmax = __riscv_vsetvlmax_e32m1();

  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
  vfloat32m1_t vec_r = __riscv_vfmv_v_f_f32m1(0, vlmax);

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, sizeof(float) * incx, vl);
    vfloat32m1_t vec_y = __riscv_vlse32_v_f32m1(y, sizeof(float) * incy, vl);
    vec_r = __riscv_vfmacc_vv_f32m1(vec_r, vec_x, vec_y, vl);
      
    x += vl;
    y += vl;
    n -= vl;
  }
  vec_r = __riscv_vfredusum_vs_f32m1_f32m1(vec_r, vec_zero, vlmax);
  *result = __riscv_vfmv_f_s_f32m1_f32(vec_r);
}


#endif
