/**
 * @file nn.h
 * @brief The Baremetal-NN Library
 * 
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

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


/**
 * Tensor0D_F16
 * 
 * A 0D tensor (scalar) with a float16_t data type.
 */
typedef struct {
  float16_t data;
} Tensor0D_F16;

/**
 * Tensor0D_F32
 * 
 * A 0D tensor (scalar) with a float data type.
 */
typedef struct {
  float data;
} Tensor0D_F32;

/**
 * Tensor1D_F16
 * 
 * A 1D tensor with a float16_t data type.
 */
typedef struct {
  size_t shape[1];
  float16_t *data;
} Tensor1D_F16;

/**
 * Tensor1D_F32
 * 
 * A 1D tensor with a float data type.
 */
typedef struct {
  size_t shape[1];
  float *data;
} Tensor1D_F32;

/**
 * Tensor2D_F16
 * 
 * A 2D tensor with a float16_t data type.
 */
typedef struct {
  size_t shape[2];
  float16_t *data;
} Tensor2D_F16;

/**
 * Tensor2D_F32
 * 
 * A 2D tensor with a float data type.
 */
typedef struct {
  size_t shape[2];
  float *data;
} Tensor2D_F32;

/**
 * Tensor3D_F16
 * 
 * A 3D tensor with a float16_t data type.
 */
typedef struct {
  size_t shape[3];
  float16_t *data;
} Tensor3D_F16;

/**
 * Tensor3D_F32
 * 
 * A 3D tensor with a float data type.
 */
typedef struct {
  size_t shape[3];
  float *data;
} Tensor3D_F32;

/**
 * Tensor4D_F16
 * 
 * A 4D tensor with a float16_t data type.
 */
typedef struct {
  size_t shape[4];
  float16_t *data;
} Tensor4D_F16;

/**
 * Tensor4D_F32
 * 
 * A 4D tensor with a float data type.
 */
typedef struct {
  size_t shape[4];
  float *data;
} Tensor4D_F32;



/**
 * nn_assert
 * 
 * Asserts that a condition is true. If the condition is false, it prints an error message and exits.
 * 
 * @param condition The condition to assert.
 * @param message The error message to print if the condition is false.
 */
static inline void nn_assert(int condition, char *message) {
  if (!condition) {
    printf("Assertion failed: ");
    printf("%s\n", message);
    exit(1);
  }
}

/**
 * float_equal
 * 
 * Checks if two floating-point numbers are equal within a relative error.
 * 
 * @param golden The expected value.
 * @param actual The actual value.
 * @param rel_err The relative error tolerance.
 * @return 1 if the numbers are equal within the relative error, 0 otherwise.
 */
static inline uint8_t float_equal(float golden, float actual, float rel_err) {
  return (fabs(actual - golden) < rel_err) || (fabs((actual - golden) / actual) < rel_err);
}


/**
 * nn_tensor0d_f16
 * 
 * Creates a 0D tensor with type F16.
 * 
 * @param data The data to store in the tensor.
 */
Tensor0D_F16 *nn_tensor0d_f16(float16_t data);

/**
 * nn_tensor0d_f32
 * 
 * Creates a 0D tensor with type F32.
 * 
 * @param data The data to store in the tensor.
 */
Tensor0D_F32 *nn_tensor0d_f32(float data);

/**
 * nn_tensor1d_f16
 * 
 * Creates a 1D tensor with type F16.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor1D_F16 *nn_tensor1d_f16(size_t shape[1], const float16_t *data);

/**
 * nn_tensor1d_f32
 * 
 * Creates a 1D tensor with type F32.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor1D_F32 *nn_tensor1d_f32(size_t shape[1], const float *data);

/**
 * nn_tensor2d_f16
 * 
 * Creates a 2D tensor with type F16.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor2D_F16 *nn_tensor2d_f16(size_t shape[2], const float16_t *data);

/**
 * nn_tensor2d_f32
 * 
 * Creates a 2D tensor with type F32.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor2D_F32 *nn_tensor2d_f32(size_t shape[2], const float *data);


/**
 * nn_print_u8
 * 
 * Prints an unsigned 8-bit integer.
 * 
 * @param v The unsigned 8-bit integer to print.
 */
void nn_print_u8(uint8_t v);

/**
 * nn_print_i8
 * 
 * Prints an 8-bit integer.
 * 
 * @param v The 8-bit integer to print.
 */
void nn_print_i8(int8_t v);

/**
 * nn_print_u16
 * 
 * Prints an unsigned 16-bit integer.
 * 
 * @param v The unsigned 16-bit integer to print.
 */
void nn_print_u16(uint16_t v);

/**
 * nn_print_i16
 * 
 * Prints an 16-bit integer.
 * 
 * @param v The 16-bit integer to print.
 */
void nn_print_i16(int16_t v);

/**
 * nn_print_u32
 * 
 * Prints an unsigned 32-bit integer.
 * 
 * @param v The unsigned 32-bit integer to print.
 */
void nn_print_u32(uint32_t v);

/**
 * nn_print_i32
 * 
 * Prints a 32-bit integer.
 * 
 * @param v The 32-bit integer to print.
 */
void nn_print_i32(int32_t v);

/**
 * nn_print_f16
 * 
 * Prints a float16_t.
 * 
 * @param v The float16_t to print.
 * @param num_digits The number of decimal digits to print.
 */
void nn_print_f16(float16_t v, int16_t num_digits);

/**
 * nn_print_f32
 * 
 * Prints a float.
 * 
 * @param v The float to print.
 * @param num_digits The number of decimal digits to print.
 */
void nn_print_f32(float v, int16_t num_digits);

/**
 * nn_print_shape
 * 
 * Prints the shape of the tensor.
 * 
 * @param ndim The number of dimensions.
 * @param shape The shape to print.
 */
void nn_print_shape(size_t ndim, const size_t *shape);

/**
 * nn_print_tensor1d_f16
 * 
 * Prints the content of a 1D tensor with type F16.
 * 
 * @param tensor The 1D tensor to print.
 */
void nn_print_tensor1d_f16(const Tensor1D_F16 *tensor);

/**
 * nn_print_tensor1d_f32
 * 
 * Prints the content of a 1D tensor with type F32.
 * 
 * @param tensor The 1D tensor to print.
 */
void nn_print_tensor1d_f32(const Tensor1D_F32 *tensor);

/**
 * nn_print_tensor2d_f16
 * 
 * Prints the content of a 2D tensor with type F16.
 * 
 * @param tensor The 2D tensor to print.
 */
void nn_print_tensor2d_f16(const Tensor2D_F16 *tensor);

/**
 * nn_print_tensor2d_f32
 * 
 * Prints the content of a 2D tensor with type F32.
 * 
 * @param tensor The 2D tensor to print.
 */
void nn_print_tensor2d_f32(const Tensor2D_F32 *tensor);

/**
 * nn_equals0d_f16
 * 
 * Checks if two 0D tensors with type F16 are equal.
 * 
 * @param a The first 0D tensor.
 * @param b The second 0D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals0d_f16(const Tensor0D_F16 *a, const Tensor0D_F16 *b, float rel_err);

/**
 * nn_equals0d_f32
 * 
 * Checks if two 0D tensors with type F32 are equal.
 * 
 * @param a The first 0D tensor.
 * @param b The second 0D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals0d_f32(const Tensor0D_F32 *a, const Tensor0D_F32 *b, float rel_err);

/**
 * nn_equals1d_f16
 * 
 * Checks if two 1D tensors with type F16 are equal.
 * 
 * @param a The first 1D tensor.
 * @param b The second 1D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals1d_f16(const Tensor1D_F16 *a, const Tensor1D_F16 *b, float rel_err);

/**
 * nn_equals1d_f32
 * 
 * Checks if two 1D tensors with type F32 are equal.
 * 
 * @param a The first 1D tensor.
 * @param b The second 1D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals1d_f32(const Tensor1D_F32 *a, const Tensor1D_F32 *b, float rel_err);

/**
 * nn_equals2d_f16
 * 
 * Checks if two 2D tensors with type F16 are equal.
 * 
 * @param a The first 2D tensor.
 * @param b The second 2D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals2d_f16(const Tensor2D_F16 *a, const Tensor2D_F16 *b, float rel_err);

/**
 * nn_equals2d_f32
 * 
 * Checks if two 2D tensors with type F32 are equal.
 * 
 * @param a The first 2D tensor.
 * @param b The second 2D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals2d_f32(const Tensor2D_F32 *a, const Tensor2D_F32 *b, float rel_err);


/**
 * nn_equals3d_f16
 * 
 * Checks if two 3D tensors with type F16 are equal.
 * 
 * @param a The first 3D tensor.
 * @param b The second 3D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals3d_f16(const Tensor3D_F16 *a, const Tensor3D_F16 *b, float rel_err);

/**
 * nn_equals3d_f32
 * 
 * Checks if two 3D tensors with type F32 are equal.
 * 
 * @param a The first 3D tensor.
 * @param b The second 3D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals3d_f32(const Tensor3D_F32 *a, const Tensor3D_F32 *b, float rel_err);

/**
 * nn_equals4d_f16
 * 
 * Checks if two 4D tensors with type F16 are equal.
 * 
 * @param a The first 4D tensor.
 * @param b The second 4D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals4d_f16(const Tensor4D_F16 *a, const Tensor4D_F16 *b, float rel_err);

/**
 * nn_equals4d_f32
 * 
 * Checks if two 4D tensors with type F32 are equal.
 * 
 * @param a The first 4D tensor.
 * @param b The second 4D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals4d_f32(const Tensor4D_F32 *a, const Tensor4D_F32 *b, float rel_err);


/**
 * nn_add1d_f16
 * 
 * Adds x1 and x2 element-wise and stores the result in y.
 * 
 * y[i] = x1[i] + x2[i]
 * 
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor.
 */
void nn_add1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x1, const Tensor1D_F16 *x2);

/**
 * nn_add1d_f32
 * 
 * Adds x1 and x2 element-wise and stores the result in y.
 * 
 * y[i] = x1[i] + x2[i]
 * 
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor. 
 */
void nn_add1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2);

/**
 * nn_add2d_f16
 * 
 * Adds x1 and x2 element-wise and stores the result in y.
 * 
 * y[i][j] = x1[i][j] + x2[i][j]
 * 
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor. 
 */
void nn_add2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2);

/**
 * nn_add2d_f32
 * 
 * Adds x1 and x2 element-wise and stores the result in y.
 * 
 * y[i][j] = x1[i][j] + x2[i][j]
 * 
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor. 
 */
void nn_add2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2);


void nn_addscalar1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x, float16_t scalar);

void nn_addscalar1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, float scalar);

void nn_addscalar2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float16_t scalar);

void nn_addscalar2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float scalar);


void nn_addmm_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, const Tensor2D_F16 *weight, const Tensor1D_F16 *bias);

void nn_addmm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const Tensor2D_F32 *weight, const Tensor1D_F32 *bias);



void nn_elu2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float alpha);

void nn_elu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float alpha);



void nn_relu2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x);

void nn_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x);


void nn_tanh2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x);

void nn_tanh2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x);


void nn_max1d_f16(Tensor0D_F16 *y, const Tensor1D_F16 *x);

void nn_max1d_f32(Tensor0D_F32 *y, const Tensor1D_F32 *x);

void nn_max2d_f16(Tensor0D_F16 *y, const Tensor2D_F16 *x);

void nn_max2d_f32(Tensor0D_F32 *y, const Tensor2D_F32 *x);


void nn_min1d_f16(Tensor0D_F16 *y, const Tensor1D_F16 *x);

void nn_min1d_f32(Tensor0D_F32 *y, const Tensor1D_F32 *x);

void nn_min2d_f16(Tensor0D_F16 *y, const Tensor2D_F16 *x);

void nn_min2d_f32(Tensor0D_F32 *y, const Tensor2D_F32 *x);


void nn_mm_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2);

void nn_mm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2);


void nn_mul1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x1, const Tensor1D_F16 *x2);

void nn_mul1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2);

void nn_mul2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2);

void nn_mul2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2);

void nn_mulscalar1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x, float16_t scalar);

void nn_mulscalar1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, float scalar);

void nn_mulscalar2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float16_t scalar);

void nn_mulscalar2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float scalar);








#endif // __NN_H