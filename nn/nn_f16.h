/**
 * @file nn.h
 * @brief The Baremetal-NN Library
 * 
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

#ifndef __NN_F16_H
#define __NN_F16_H

#include "float16.h"


/**
 * Tensor0D_F16
 * 
 * A 0D tensor (scalar) with a half-precision floating-point data type.
 */
typedef struct {
  float16_t data;
} Tensor0D_F16;


/**
 * Tensor1D_F16
 * 
 * A 1D tensor with a half-precision floating-point data type.
 */
typedef struct {
  size_t shape[1];
  float16_t *data;
} Tensor1D_F16;


/**
 * Tensor2D_F16
 * 
 * A 2D tensor with a half-precision floating-point data type.
 */
typedef struct {
  size_t shape[2];
  float16_t *data;
} Tensor2D_F16;

/**
 * nn_equal_f32
 * 
 * Checks if two half-precision floating-point numbers are equal within a relative error.
 * 
 * @param golden The expected value.
 * @param actual The actual value.
 * @param rel_err The relative error tolerance.
 * @return 1 if the numbers are equal within the relative error, 0 otherwise.
 */
static inline uint8_t nn_equal_f16(float16_t golden, float16_t actual, float rel_err) {
  return (fabs(as_f32(actual) - as_f32(golden)) < rel_err) || (fabs((as_f32(actual) - as_f32(golden)) / as_f32(actual)) < rel_err);
}


/* ======================================================================================================== */
/*                                           Tensor Creation                                                */
/* ======================================================================================================== */
/**
 * nn_tensor0d_f16
 * 
 * Creates a 0D tensor with type F16.
 * 
 * @param data The data to store in the tensor.
 */
Tensor0D_F16 *nn_tensor0d_f16(float16_t data) {
  Tensor0D_F16 *tensor = (Tensor0D_F16 *)malloc(sizeof(Tensor0D_F16));
  tensor->data = data;
}

/**
 * nn_tensor1d_f16
 * 
 * Creates a 1D tensor with type F16.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor1D_F16 *nn_tensor1d_f16(size_t shape[1], const float16_t *data) {
  Tensor1D_F16 *tensor = (Tensor1D_F16 *)malloc(sizeof(Tensor1D_F16));
  tensor->shape[0] = shape[0];

  size_t n_bytes = shape[0] * sizeof(float16_t);
  tensor->data = (float16_t *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
}

/**
 * nn_tensor2d_f16
 * 
 * Creates a 2D tensor with type F16.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor2D_F16 *nn_tensor2d_f16(size_t shape[2], const float16_t *data) {
  Tensor2D_F16 *tensor = (Tensor2D_F16 *)malloc(sizeof(Tensor2D_F16));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];

  size_t n_bytes = shape[0] * shape[1] * sizeof(float16_t);
  tensor->data = (float16_t *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
}

Tensor0D_F16 *nn_zeros0d_f16() {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(0);
  return tensor;
}

Tensor1D_F16 *nn_zeros1d_f16(size_t shape[1]) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

Tensor2D_F16 *nn_zeros2d_f16(size_t shape[2]) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

Tensor0D_F16 *nn_ones0d_f16() {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(1);
  return tensor;
}

Tensor1D_F16 *nn_ones1d_f16(size_t shape[1]) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

Tensor2D_F16 *nn_ones2d_f16(size_t shape[2]) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

Tensor0D_F16 *nn_full0d_f16(float16_t data) {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(data);
  return tensor;
}

Tensor1D_F16 *nn_full1d_f16(size_t shape[1], float16_t data) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

Tensor2D_F16 *nn_full2d_f16(size_t shape[2], float16_t data) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

Tensor0D_F16 *nn_rand0d_f16() {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(as_f16(rand()));
  return tensor;
}

Tensor1D_F16 *nn_rand1d_f16(size_t shape[1]) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = as_f16(rand());
  }
  return tensor;
}

Tensor2D_F16 *nn_rand2d_f16(size_t shape[2]) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = as_f16(rand());
  }
  return tensor;
}


/* ======================================================================================================== */
/*                                           Tensor Prints                                                  */
/* ======================================================================================================== */
/**
 * nn_print_f16
 * 
 * Prints a float.
 * 
 * @param v The float to print.
 * @param num_digits The number of decimal digits to print.
 */
void nn_print_f16(float16_t v, int16_t num_digits) {
  nn_print_f32(as_f32(v), num_digits);
}


/**
 * nn_print_tensor1d_f16
 * 
 * Prints the content of a 1D tensor with type F16.
 * 
 * @param tensor The 1D tensor to print.
 */

void nn_print_tensor1d_f16(const Tensor1D_F16 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    nn_print_f16(*((float16_t *)tensor->data + i), 3);
    if (i < tensor->shape[0]-1) {
      printf(" ");
    }
  }
  printf("]\n");
}

/**
 * nn_print_tensor2d_f16
 * 
 * Prints the content of a 2D tensor with type F16.
 * 
 * @param tensor The 2D tensor to print.
 */
void nn_print_tensor2d_f16(const Tensor2D_F16 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i != 0) {
      printf(" ");
    }
    printf("[");
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      nn_print_f16(*((float16_t *)tensor->data + i*tensor->shape[1] + j), 3);
      if (j < tensor->shape[1]-1) {
        printf(" ");
      }
    }
    printf("]");
    if (i < tensor->shape[0]-1) {
      printf("\n");
    }
  }
  printf("]\n");
}


/* ======================================================================================================== */
/*                                           Comparision                                                    */
/* ======================================================================================================== */
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
uint8_t nn_equals0d_f16(const Tensor0D_F16 *a, const Tensor0D_F16 *b, float rel_err) {
  return nn_equal_f16(a->data, b->data, rel_err);
}

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
uint8_t nn_equals1d_f16(const Tensor1D_F16 *a, const Tensor1D_F16 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    if (!nn_equal_f16(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}

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
uint8_t nn_equals2d_f16(const Tensor2D_F16 *a, const Tensor2D_F16 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (!nn_equal_f16(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}



/* ======================================================================================================== */
/*                                           Unary                                                          */
/* ======================================================================================================== */
void nn_max1d_f16(Tensor0D_F16 *y, const Tensor1D_F16 *x) {
  y->data = -FLT_MAX;
  size_t n = x->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    float val = as_f32(x->data[i]);
    y->data = val > y->data ? val : y->data;
  }
}

void nn_max2d_f16(Tensor0D_F16 *y, const Tensor2D_F16 *x) {
  y->data = -FLT_MAX;
  size_t n = x->shape[0] * x->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float val = as_f32(x->data[i]);
    y->data = val > y->data ? val : y->data;
  }
}


void nn_min1d_f16(Tensor0D_F16 *y, const Tensor1D_F16 *x) {
  y->data = FLT_MAX;
  size_t n = x->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    float val = as_f32(x->data[i]);
    y->data = val < y->data ? val : y->data;
  }
}


void nn_min2d_f16(Tensor0D_F16 *y, const Tensor2D_F16 *x) {
  y->data = FLT_MAX;
  size_t n = x->shape[0] * x->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float val = as_f32(x->data[i]);
    y->data = val < y->data ? val : y->data;
  }
}


/* ======================================================================================================== */
/*                                           Addition                                                       */
/* ======================================================================================================== */
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
void nn_add1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x1, const Tensor1D_F16 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x1->data[i]) + as_f32(x2->data[i]));
  }
}



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
void nn_add2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x1->data[i]) + as_f32(x2->data[i]));
  }
}

void nn_addscalar1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x->data[i]) + as_f32(scalar));
  }
}

void nn_addscalar2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x->data[i]) + as_f32(scalar));
  }
}




/* ======================================================================================================== */
/*                                           Multiplication                                                 */
/* ======================================================================================================== */


void nn_mul1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x1, const Tensor1D_F16 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x1->data[i]) * as_f32(x2->data[i]));
  }
}


void nn_mul2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x1->data[i]) * as_f32(x2->data[i]));
  }
}


void nn_mulscalar1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x->data[i]) * as_f32(scalar));
  }
}



void nn_mulscalar2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(as_f32(x->data[i]) * as_f32(scalar));
  }
}



/* ======================================================================================================== */
/*                                           MatMul                                                         */
/* ======================================================================================================== */
void nn_dot_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x1, const Tensor1D_F16 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot dot tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot dot tensors of different shapes");

  size_t n = y->shape[0];
  float sum_f32 = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum_f32 += as_f32(x1->data[i]) * as_f32(x2->data[i]);
  }
  y->data[0] = as_f16(sum_f32);
}


void nn_mm_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) { 
  nn_assert(x1->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");

  const size_t batch_size = x1->shape[0];
  const size_t in_features = x1->shape[1];
  const size_t out_features = x2->shape[0];

  for (size_t i = 0; i < batch_size; i += 1) {
    for (size_t j = 0; j < out_features; j += 1) {
      float sum = 0.f;
      for (size_t k = 0; k < in_features; k += 1) {
        sum += as_f32(x1->data[i * in_features + k]) * as_f32(x2->data[j * in_features + k]);
      }
      y->data[i * out_features + j] = as_f16(sum);
    }
  }
}



void nn_addmm_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, const Tensor2D_F16 *weight, const Tensor1D_F16 *bias) { 
  nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  nn_assert(bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < out_features; j++) {
      float sum = 0.f;
      for (size_t k = 0; k < in_features; k++) {
        sum += as_f32(x->data[i * in_features + k]) * as_f32(weight->data[j * in_features + k]);
      }
      y->data[i * out_features + j] = as_f16(sum + as_f32(bias->data[j]));
    }
  }
}



/* ======================================================================================================== */
/*                                           Non-linear                                                     */
/* ======================================================================================================== */
void nn_elu2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float alpha) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ELU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (as_f32(x->data[i]) > 0) {
      y->data[i] = x->data[i];
    }
    else {
      y->data[i] = as_f16(alpha * (expf(as_f32(x->data[i])) - 1.f));
    }
  }
}


void nn_relu2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = as_f32(x->data[i]);
    y->data[i] = x_val > 0 ? as_f16(x_val) : 0;
  }
}


void nn_softmax1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float sum = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    sum += expf(as_f32(x->data[i]));
  }

  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = as_f16(expf(as_f32(x->data[i])) / sum);
  }
}



void nn_tanh2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = as_f32(x->data[i]);
    y->data[i] = as_f16(tanh(x_val));
  }
}





#endif // __NN_F32_H