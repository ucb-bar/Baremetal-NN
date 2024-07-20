#include <riscv_vector.h>
#include "impl/minimum1.h"

#ifdef RVV

void NN__minimum1_i8(size_t n, int8_t *y, size_t incy, const int8_t *x, size_t incx, int8_t scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vint8m1_t vec_x = __riscv_vlse8_v_i8m1(x, sizeof(int8_t) * incx, vl);
    vint8m1_t vec_s = __riscv_vmv_v_x_i8m1(scalar, vl);
    vint8m1_t vec_y = __riscv_vmin_vv_i8m1(vec_x, vec_s, vl);
    __riscv_vsse8_v_i8m1(y, sizeof(int8_t) * incy, vec_y, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}

void NN__minimum1_i16(size_t n, int16_t *y, size_t incy, const int16_t *x, size_t incx, int16_t scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e16m1(n);
    vint16m1_t vec_x = __riscv_vlse16_v_i16m1(x, sizeof(int16_t) * incx, vl);
    vint16m1_t vec_s = __riscv_vmv_v_x_i16m1(scalar, vl);
    vint16m1_t vec_y = __riscv_vmin_vv_i16m1(vec_x, vec_s, vl);
    __riscv_vsse16_v_i16m1(y, sizeof(int16_t) * incy, vec_y, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}

void NN__minimum1_i32(size_t n, int32_t *y, size_t incy, const int32_t *x, size_t incx, int32_t scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vint32m1_t vec_x = __riscv_vlse32_v_i32m1(x, sizeof(int32_t) * incx, vl);
    vint32m1_t vec_s = __riscv_vmv_v_x_i32m1(scalar, vl);
    vint32m1_t vec_y = __riscv_vmin_vv_i32m1(vec_x, vec_s, vl);
    __riscv_vsse32_v_i32m1(y, sizeof(int32_t) * incy, vec_y, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}

#ifdef ZVFH
  void NN__minimum1_f16(size_t n, float16_t *y, size_t incy, const float16_t *x, size_t incx, float16_t scalar) {
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vlse16_v_f16m1(x, sizeof(float16_t) * incx, vl);
      vfloat16m1_t vec_s = __riscv_vfmv_v_f_f16m1(scalar, vl);
      vfloat16m1_t vec_y = __riscv_vfmin_vv_f16m1(vec_x, vec_s, vl);
      __riscv_vsse16_v_f16m1(y, sizeof(float16_t) * incy, vec_y, vl);
      x += vl;
      y += vl;
      n -= vl;
    }
  }
#endif

void NN__minimum1_f32(size_t n, float *y, size_t incy, const float *x, size_t incx, float scalar) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x, sizeof(float) * incx, vl);
    vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(scalar, vl);
    vfloat32m1_t vec_y = __riscv_vfmin_vv_f32m1(vec_x, vec_s, vl);
    __riscv_vsse32_v_f32m1(y, sizeof(float) * incy, vec_y, vl);
    x += vl;
    y += vl;
    n -= vl;
  }
}


#endif
