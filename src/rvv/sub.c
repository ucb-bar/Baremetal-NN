#include <riscv_vector.h>
#include "ops/sub.h"

#ifdef RVV

void NN_sub_u8(size_t n, uint8_t *y, size_t incy, const uint8_t *x1, size_t incx1, const uint8_t *x2, size_t incx2) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vuint8m1_t vec_x1 = __riscv_vlse8_v_u8m1(x1, sizeof(uint8_t) * incx1, vl);
    vuint8m1_t vec_x2 = __riscv_vlse8_v_u8m1(x2, sizeof(uint8_t) * incx2, vl);
    vuint8m1_t vec_y = __riscv_vsub_vv_u8m1(vec_x1, vec_x2, vl);
    __riscv_vsse8_v_u8m1(y, sizeof(uint8_t) * incy, vec_y, vl);
    x1 += vl;
    x2 += vl;
    y += vl;
    n -= vl;
  }
}

void NN_sub_i8(size_t n, int8_t *y, size_t incy, const int8_t *x1, size_t incx1, const int8_t *x2, size_t incx2) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    vint8m1_t vec_x1 = __riscv_vlse8_v_i8m1(x1, sizeof(int8_t) * incx1, vl);
    vint8m1_t vec_x2 = __riscv_vlse8_v_i8m1(x2, sizeof(int8_t) * incx2, vl);
    vint8m1_t vec_y = __riscv_vsub_vv_i8m1(vec_x1, vec_x2, vl);
    __riscv_vsse8_v_i8m1(y, sizeof(int8_t) * incy, vec_y, vl);
    x1 += vl;
    x2 += vl;
    y += vl;
    n -= vl;
  }
}

void NN_sub_i16(size_t n, int16_t *y, size_t incy, const int16_t *x1, size_t incx1, const int16_t *x2, size_t incx2) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e16m1(n);
    vint16m1_t vec_x1 = __riscv_vlse16_v_i16m1(x1, sizeof(int16_t) * incx1, vl);
    vint16m1_t vec_x2 = __riscv_vlse16_v_i16m1(x2, sizeof(int16_t) * incx2, vl);
    vint16m1_t vec_y = __riscv_vsub_vv_i16m1(vec_x1, vec_x2, vl);
    __riscv_vsse16_v_i16m1(y, sizeof(int16_t) * incy, vec_y, vl);
    x1 += vl;
    x2 += vl;
    y += vl;
    n -= vl;
  }
}

void NN_sub_i32(size_t n, int32_t *y, size_t incy, const int32_t *x1, size_t incx1, const int32_t *x2, size_t incx2) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vint32m1_t vec_x1 = __riscv_vlse32_v_i32m1(x1, sizeof(int32_t) * incx1, vl);
    vint32m1_t vec_x2 = __riscv_vlse32_v_i32m1(x2, sizeof(int32_t) * incx2, vl);
    vint32m1_t vec_y = __riscv_vsub_vv_i32m1(vec_x1, vec_x2, vl);
    __riscv_vsse32_v_i32m1(y, sizeof(int32_t) * incy, vec_y, vl);
    x1 += vl;
    x2 += vl;
    y += vl;
    n -= vl;
  }
}

#ifdef RISCV_ZVFH
  void NN_sub_f16(size_t n, float16_t *y, size_t incy, const float16_t *x1, size_t incx1, const float16_t *x2, size_t incx2) {
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x1 = __riscv_vlse16_v_f16m1(x1, sizeof(float16_t) * incx1, vl);
      vfloat16m1_t vec_x2 = __riscv_vlse16_v_f16m1(x2, sizeof(float16_t) * incx2, vl);
      vfloat16m1_t vec_y = __riscv_vfsub_vv_f16m1(vec_x1, vec_x2, vl);
      __riscv_vsse16_v_f16m1(y, sizeof(float16_t) * incy, vec_y, vl);
      x1 += vl;
      x2 += vl;
      y += vl;
      n -= vl;
    }
  }
#endif

void NN_sub_f32(size_t n, float *y, size_t incy, const float *x1, size_t incx1, const float *x2, size_t incx2) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vec_x1 = __riscv_vlse32_v_f32m1(x1, sizeof(float) * incx1, vl);
    vfloat32m1_t vec_x2 = __riscv_vlse32_v_f32m1(x2, sizeof(float) * incx2, vl);
    vfloat32m1_t vec_y = __riscv_vfsub_vv_f32m1(vec_x1, vec_x2, vl);
    __riscv_vsse32_v_f32m1(y, sizeof(float) * incy, vec_y, vl);
    x1 += vl;
    x2 += vl;
    y += vl;
    n -= vl;
  }
}

#endif
