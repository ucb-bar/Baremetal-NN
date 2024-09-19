#ifndef __NN_H
#define __NN_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "float16.h"


// http://elm-chan.org/junk/32bit/binclude.html
#define INCLUDE_FILE(section, filename, symbol) asm (\
    ".section "#section"\n"                   /* Change section */\
    ".balign 4\n"                             /* Word alignment */\
    ".global "#symbol"_start\n"               /* Export the object start address */\
    ".global "#symbol"_data\n"                /* Export the object address */\
    #symbol"_start:\n"                        /* Define the object start address label */\
    #symbol"_data:\n"                         /* Define the object label */\
    ".incbin \""filename"\"\n"                /* Import the file */\
    ".global "#symbol"_end\n"                 /* Export the object end address */\
    #symbol"_end:\n"                          /* Define the object end address label */\
    ".balign 4\n"                             /* Word alignment */\
    ".section \".text\"\n")                   /* Restore section */



typedef struct {
  size_t shape[1];
  float16_t *data;
} Tensor1D_F16;

typedef struct {
  size_t shape[1];
  float *data;
} Tensor1D_F32;

typedef struct {
  size_t shape[2];
  float16_t *data;
} Tensor2D_F16;

typedef struct {
  size_t shape[2];
  float *data;
} Tensor2D_F32;



static inline void NN_assert(int condition, char *message) {
  if (!condition) {
    printf("Assertion failed: ");
    printf("%s\n", message);
    exit(1);
  }
}


static inline uint8_t float_equal(float golden, float actual, float rel_err) {
  return (fabs(actual - golden) < rel_err) || (fabs((actual - golden) / actual) < rel_err);
}



void NN_print_u8(uint8_t v);

void NN_print_i8(int8_t v);

void NN_print_u16(uint16_t v);

void NN_print_i16(int16_t v);

void NN_print_u32(uint32_t v);

void NN_print_i32(int32_t v);

void NN_print_f16(float16_t v, int16_t num_digits);

void NN_print_f32(float v, int16_t num_digits);

void NN_print_shape(size_t ndim, const size_t *shape);

void NN_print_tensor1d_f16(const Tensor1D_F16 *tensor);

void NN_print_tensor1d_f32(const Tensor1D_F32 *tensor);

void NN_print_tensor2d_f16(const Tensor2D_F16 *tensor);

void NN_print_tensor2d_f32(const Tensor2D_F32 *tensor);

// void NN_print_tensor3d_f16(const Tensor3D_F16 *tensor);

// void NN_print_tensor3d_f32(const Tensor3D_F32 *tensor);

// void NN_print_tensor4d_f16(const Tensor4D_F16 *tensor);

// void NN_print_tensor4d_f32(const Tensor4D_F32 *tensor);


uint8_t NN_equals1d_f32(const Tensor1D_F32 *a, const Tensor1D_F32 *b, float rel_err);

uint8_t NN_equals2d_f32(const Tensor2D_F32 *a, const Tensor2D_F32 *b, float rel_err);


void NN_add1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2);
void NN_add2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2);


void NN_linear_f16(size_t batch_size, size_t in_features, size_t out_features, float16_t *out, const float16_t *input, const float16_t *weight, const float16_t *bias);
void NN_linear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const Tensor2D_F32 *weight, const Tensor1D_F32 *bias);

void NN_elu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float alpha);
void NN_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x);


#endif // __NN_H