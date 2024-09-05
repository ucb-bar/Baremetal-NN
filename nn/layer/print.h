#ifndef __NN_PRINT_H
#define __NN_PRINT_H

#include <math.h>

#include "tensor.h"

static inline void NN_print_u8(uint8_t v) {
  printf("%d", v);
}

static inline void NN_print_i8(int8_t v) {
  printf("%d", v);
}

static inline void NN_print_u16(uint16_t v) {
  printf("%d", v);
}

static inline void NN_print_i16(int16_t v) {
  printf("%d", v);
}

static inline void NN_print_u32(uint32_t v) {
  printf("%ld", (size_t)v);
}

static inline void NN_print_i32(int32_t v) {
  printf("%ld", (size_t)v);
}

void NN_print_f16(float16_t v, int16_t num_digits);

void NN_print_f32(float v, int16_t num_digits);

void NN_print_shape(const Tensor *t);

void NN_printf(const Tensor *t);


#endif // __NN_PRINT_H
