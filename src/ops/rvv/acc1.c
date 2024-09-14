#include <riscv_vector.h>
#include "ops/acc1.h"

#ifdef RVV

void NN_acc1_i8(size_t n, int8_t *r, size_t incx, int8_t s) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vint8m1_t vec_r = __riscv_vlse8_v_i8m1(r, sizeof(int8_t) * incx, vl);
    vint8m1_t vec_s = __riscv_vmv_v_x_i8m1(s, vl);
    vec_r = __riscv_vadd_vv_i8m1(vec_r, vec_s, vl);
    __riscv_vse8_v_i8m1(r, vec_r, vl);
    r += vl;
    n -= vl;
  }
}

void NN_acc1_i16(size_t n, int16_t *r, size_t incx, int16_t s) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e16m1(n);
    vint16m1_t vec_r = __riscv_vlse16_v_i16m1(r, sizeof(int16_t) * incx, vl);
    vint16m1_t vec_s = __riscv_vmv_v_x_i16m1(s, vl);
    vec_r = __riscv_vadd_vv_i16m1(vec_r, vec_s, vl);
    __riscv_vse16_v_i16m1(r, vec_r, vl);
    r += vl;
    n -= vl;
  }
}

void NN_acc1_i32(size_t n, int32_t *r, size_t incx, int32_t s) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vint32m1_t vec_r = __riscv_vlse32_v_i32m1(r, sizeof(int32_t) * incx, vl);
    vint32m1_t vec_s = __riscv_vmv_v_x_i32m1(s, vl);
    vec_r = __riscv_vadd_vv_i32m1(vec_r, vec_s, vl);
    __riscv_vse32_v_i32m1(r, vec_r, vl);
    r += vl;
    n -= vl;
  }
}

#ifdef RISCV_ZVFH
  void NN_acc1_f16(size_t n, float16_t *r, size_t incx, float16_t s) {
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_r = __riscv_vlse16_v_f16m1(r, sizeof(float16_t) * incx, vl);
      vfloat16m1_t vec_s = __riscv_vfmv_v_f_f16m1(s, vl);
      vec_r = __riscv_vfadd_vv_f16m1(vec_r, vec_s, vl);
      __riscv_vse16_v_f16m1(r, vec_r, vl);
      r += vl;
      n -= vl;
    }
  }
#endif

void NN_acc1_f32(size_t n, float *r, size_t incx, float s) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_r = __riscv_vlse32_v_f32m1(r, sizeof(float) * incx, vl);
    vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(s, vl);
    vec_r = __riscv_vfadd_vv_f32m1(vec_r, vec_s, vl);
    __riscv_vse32_v_f32m1(r, vec_r, vl);
    r += vl;
    n -= vl;
  }
}

#endif