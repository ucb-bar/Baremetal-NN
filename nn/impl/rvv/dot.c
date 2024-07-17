#include <riscv_vector.h>
#include "dot.h"

#ifdef RVV

void NN__dot_i8_to_i32(size_t n, int32_t *result, const int8_t *x, size_t incx, const int8_t *y, size_t incy) {
  size_t vlmax = __riscv_vsetvlmax_e8m1();

  vint16m1_t vec_r = __riscv_vmv_v_x_i16m1(0, vlmax);
  vint16m2_t vec_r_m2 = __riscv_vmv_v_x_i16m2(0, vlmax);

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vint8m1_t vec_x = __riscv_vlse8_v_i8m1(x, sizeof(int8_t) * incx, vl);
    vint8m1_t vec_y = __riscv_vlse8_v_i8m1(y, sizeof(int8_t) * incy, vl);
    vec_r_m2 = __riscv_vwmacc_vv_i16m2(vec_r_m2, vec_x, vec_y, vl);

    x += vl;
    y += vl;
    n -= vl;
  }
  vec_r = __riscv_vredsum_vs_i16m2_i16m1(vec_r_m2, vec_r, vlmax);
  *result = __riscv_vmv_x_s_i16m1_i16(vec_r);
}

void NN__dot_i16_to_i32(size_t n, int32_t *result, const int16_t *x, size_t incx, const int16_t *y, size_t incy) {
  size_t vlmax = __riscv_vsetvlmax_e16m1();

  vint32m1_t vec_r = __riscv_vmv_v_x_i32m1(0, vlmax);
  vint32m2_t vec_r_m2 = __riscv_vmv_v_x_i32m2(0, vlmax);

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e16m1(n);
    vint16m1_t vec_x = __riscv_vlse16_v_i16m1(x, sizeof(int16_t) * incx, vl);
    vint16m1_t vec_y = __riscv_vlse16_v_i16m1(y, sizeof(int16_t) * incy, vl);
    vec_r_m2 = __riscv_vwmacc_vv_i32m2(vec_r_m2, vec_x, vec_y, vl);     

    x += vl;
    y += vl;
    n -= vl;
  }
  vec_r = __riscv_vredsum_vs_i32m2_i32m1(vec_r_m2, vec_r, vlmax);
  *result = __riscv_vmv_x_s_i32m1_i32(vec_r);
}

void NN__dot_i32(size_t n, int32_t *result, const int32_t *x, size_t incx, const int32_t *y, size_t incy) {
  size_t vlmax = __riscv_vsetvlmax_e32m1();

  vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
  vint32m1_t vec_r = __riscv_vmv_v_x_i32m1(0, vlmax);

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vint32m1_t vec_x = __riscv_vlse32_v_i32m1(x, sizeof(int32_t) * incx, vl);
    vint32m1_t vec_y = __riscv_vlse32_v_i32m1(y, sizeof(int32_t) * incy, vl);
    vec_r = __riscv_vmacc_vv_i32m1(vec_r, vec_x, vec_y, vl);     

    x += vl;
    y += vl;
    n -= vl;
  }
  vec_r = __riscv_vredsum_vs_i32m1_i32m1(vec_r, vec_zero, vlmax);
  *result = __riscv_vmv_x_s_i32m1_i32(vec_r);
}

// void NN__dot_f16(size_t n, float16_t *result, const float16_t *x, size_t incx, const float16_t *y, size_t incy) {
//   size_t vlmax = __riscv_vsetvlmax_e16m1();

//   vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
//   vfloat16m1_t vec_r = __riscv_vfmv_v_f_f16m1(0, vlmax);

//   while (n > 0) {
//     size_t vl = __riscv_vsetvl_e16m1(n);
//     vfloat16m1_t vec_x = __riscv_vlse16_v_f16m1(x, sizeof(float16_t) * incx, vl);
//     vfloat16m1_t vec_y = __riscv_vlse16_v_f16m1(y, sizeof(float16_t) * incy, vl);
//     vec_r = __riscv_vfmacc_vv_f16m1(vec_r, vec_x, vec_y, vl);     

//     x += vl;
//     y += vl;
//     n -= vl;
//   }
//   vec_r = __riscv_vfredusum_vs_f16m1_f16m1(vec_r, vec_zero, vlmax);
//   *result = __riscv_vfmv_f_s_f16m1_f16(vec_r);
// }

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
