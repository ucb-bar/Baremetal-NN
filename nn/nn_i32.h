/**
 * @file nn.h
 * @brief The Baremetal-NN Library
 * 
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

#ifndef __NN_I32_H
#define __NN_I32_H

#include "float16.h"


#ifdef CONFIG_BACKEND_RISCV_V
  #include "riscv_vector.h"
#endif

/**
 * Tensor0D_I32
 * 
 * A 0D tensor (scalar) with a int32_t data type.
 */
typedef struct {
  int32_t data;
} Tensor0D_I32;

/**
 * Tensor1D_I32
 * 
 * A 1D tensor with a int32_t data type.
 */
typedef struct {
  size_t shape[1];
  int32_t *data;
} Tensor1D_I32;

/**
 * Tensor2D_I32
 * 
 * A 2D tensor with a int32_t data type.
 */
typedef struct {
  size_t shape[2];
  int32_t *data;
} Tensor2D_I32;


/* ======================================================================================================== */
/*                                           Tensor Creation                                                */
/* ======================================================================================================== */
/**
 * nn_tensor0d_i32
 * 
 * Creates a 0D tensor with type I32.
 * 
 * @param data The data to store in the tensor.
 */
Tensor0D_I32 *nn_tensor0d_i32(int32_t data) {
  Tensor0D_I32 *tensor = (Tensor0D_I32 *)malloc(sizeof(Tensor0D_I32));
  tensor->data = data;
  return tensor;
}

/**
 * nn_tensor1d_i32
 * 
 * Creates a 1D tensor with type I32.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor1D_I32 *nn_tensor1d_i32(size_t shape[1], const int32_t *data) {
  Tensor1D_I32 *tensor = (Tensor1D_I32 *)malloc(sizeof(Tensor1D_I32));
  tensor->shape[0] = shape[0];

  size_t n_bytes = shape[0] * sizeof(int32_t);
  tensor->data = (int32_t *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
  return tensor;
}

/**
 * nn_tensor2d_i32
 * 
 * Creates a 2D tensor with type I32.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor2D_I32 *nn_tensor2d_i32(size_t shape[2], const int32_t *data) {
  Tensor2D_I32 *tensor = (Tensor2D_I32 *)malloc(sizeof(Tensor2D_I32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];

  size_t n_bytes = shape[0] * shape[1] * sizeof(int32_t);
  tensor->data = (int32_t *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
  return tensor;
}


Tensor0D_I32 *nn_zeros0d_i32() {
  Tensor0D_I32 *tensor = nn_tensor0d_i32(0);
  return tensor;
}

Tensor1D_I32 *nn_zeros1d_i32(size_t shape[1]) {
  Tensor1D_I32 *tensor = nn_tensor1d_i32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

Tensor2D_I32 *nn_zeros2d_i32(size_t shape[2]) {
  Tensor2D_I32 *tensor = nn_tensor2d_i32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

Tensor0D_I32 *nn_ones0d_i32() {
  Tensor0D_I32 *tensor = nn_tensor0d_i32(1);
  return tensor;
}

Tensor1D_I32 *nn_ones1d_i32(size_t shape[1]) {
  Tensor1D_I32 *tensor = nn_tensor1d_i32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

Tensor2D_I32 *nn_ones2d_i32(size_t shape[2]) {
  Tensor2D_I32 *tensor = nn_tensor2d_i32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

Tensor0D_I32 *nn_full0d_i32(int32_t data) {
  Tensor0D_I32 *tensor = nn_tensor0d_i32(data);
  return tensor;
}

Tensor1D_I32 *nn_full1d_i32(size_t shape[1], int32_t data) {
  Tensor1D_I32 *tensor = nn_tensor1d_i32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

Tensor2D_I32 *nn_full2d_i32(size_t shape[2], int32_t data) {
  Tensor2D_I32 *tensor = nn_tensor2d_i32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

Tensor0D_I32 *nn_rand0d_i32() {
  Tensor0D_I32 *tensor = nn_tensor0d_i32(rand());
  return tensor;
}

Tensor1D_I32 *nn_rand1d_i32(size_t shape[1]) {
  Tensor1D_I32 *tensor = nn_tensor1d_i32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
  }
  return tensor;
}

Tensor2D_I32 *nn_rand2d_i32(size_t shape[2]) {
  Tensor2D_I32 *tensor = nn_tensor2d_i32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
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
void nn_print_i32(int32_t v, int16_t num_digits) {
  printf("%d", v);
}


/**
 * nn_print_tensor1d_f16
 * 
 * Prints the content of a 1D tensor with type F16.
 * 
 * @param tensor The 1D tensor to print.
 */

void nn_print_tensor1d_i32(const Tensor1D_I32 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    nn_print_i32(*((int32_t *)tensor->data + i), 3);
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
void nn_print_tensor2d_i32(const Tensor2D_I32 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i != 0) {
      printf(" ");
    }
    printf("[");
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      nn_print_i32(*((int32_t *)tensor->data + i*tensor->shape[1] + j), 3);
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
 * nn_equals0d_i32
 * 
 * Checks if two 0D tensors with type I32 are equal.
 * 
 * @param a The first 0D tensor.
 * @param b The second 0D tensor.
 * @return 1 if the tensors are equal, 0 otherwise.
 */
uint8_t nn_equals0d_i32(const Tensor0D_I32 *a, const Tensor0D_I32 *b) {
  return a->data == b->data;
}

/**
 * nn_equals1d_i32
 * 
 * Checks if two 1D tensors with type I32 are equal.
 * 
 * @param a The first 1D tensor.
 * @param b The second 1D tensor.
 * @return 1 if the tensors are equal, 0 otherwise.
 */
uint8_t nn_equals1d_i32(const Tensor1D_I32 *a, const Tensor1D_I32 *b) {
  nn_assert(a->shape[0] == b->shape[0], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    if (a->data[i] != b->data[i]) {
      return 0;
    }
  }
  return 1;
}

/**
 * nn_equals2d_i32
 * 
 * Checks if two 2D tensors with type I32 are equal.
 * 
 * @param a The first 2D tensor.
 * @param b The second 2D tensor.
 * @return 1 if the tensors are equal, 0 otherwise.
 */
uint8_t nn_equals2d_i32(const Tensor2D_I32 *a, const Tensor2D_I32 *b) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (a->data[i] != b->data[i]) {
      return 0;
    }
  }
  return 1;
}


/* ======================================================================================================== */
/*                                           Unary                                                          */
/* ======================================================================================================== */




/* ======================================================================================================== */
/*                                           Addition                                                       */
/* ======================================================================================================== */
/**
 * nn_add1d_i32
 * 
 * Adds x1 and x2 element-wise and stores the result in y.
 * 
 * y[i] = x1[i] + x2[i]
 * 
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor. 
 */
void nn_add1d_i32(Tensor1D_I32 *y, const Tensor1D_I32 *x1, const Tensor1D_I32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  int32_t *x1_data = x1->data;
  int32_t *x2_data = x2->data;
  int32_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_VECTOR
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m1(n);
      vint32m1_t vec_x1 = __riscv_vle32_v_i32m1(x1_data, vl);
      vint32m1_t vec_x2 = __riscv_vle32_v_i32m1(x2_data, vl);
      vint32m1_t vec_y = __riscv_vfadd_vv_i32m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_i32m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = x1->data[i] + x2->data[i];
    }
  #endif
}



/**
 * nn_add2d_i32
 * 
 * Adds x1 and x2 element-wise and stores the result in y.
 * 
 * y[i][j] = x1[i][j] + x2[i][j]
 * 
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor. 
 */
void nn_add2d_i32(Tensor2D_I32 *y, const Tensor2D_I32 *x1, const Tensor2D_I32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  int32_t *x1_data = x1->data;
  int32_t *x2_data = x2->data;
  int32_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_VECTOR
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m1(n);
      vint32m1_t vec_x1 = __riscv_vle32_v_i32m1(x1_data, vl);
      vint32m1_t vec_x2 = __riscv_vle32_v_i32m1(x2_data, vl);
      vint32m1_t vec_y = __riscv_vfadd_vv_i32m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_i32m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = x1->data[i] + x2->data[i];
    }
  #endif
}

void nn_addscalar1d_i32(Tensor1D_I32 *y, const Tensor1D_I32 *x, int32_t scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] + scalar;
  }
}

void nn_addscalar2d_i32(Tensor2D_I32 *y, const Tensor2D_I32 *x, int32_t scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];

  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] + scalar;
  }
}




/* ======================================================================================================== */
/*                                           Multiplication                                                 */
/* ======================================================================================================== */




/* ======================================================================================================== */
/*                                           MatMul                                                         */
/* ======================================================================================================== */
void nn_dot_i32(Tensor1D_I32 *y, const Tensor1D_I32 *x1, const Tensor1D_I32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot dot tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot dot tensors of different shapes");

  size_t n = y->shape[0];
  int32_t sum_i32 = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum_i32 += x1->data[i] * x2->data[i];
  }
  y->data[0] = sum_i32;
}


void nn_mm_i32(Tensor2D_I32 *y, const Tensor2D_I32 *x1, const Tensor2D_I32 *x2) { 
  nn_assert(x1->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");

  const size_t n = x1->shape[0];
  const size_t m = x1->shape[1];
  const size_t p = x2->shape[1];

  for (size_t i = 0; i < n; i += 1) {
    for (size_t j = 0; j < p; j += 1) {
      int32_t sum = 0;
      for (size_t k = 0; k < m; k += 1) {
        sum += x1->data[i * m + k] * x2->data[k * p + j];
      }
      y->data[i * p + j] = sum;
    }
  }
}

void nn_addmm_i32(Tensor2D_I32 *y, const Tensor2D_I32 *c, const Tensor2D_I32 *x1, const Tensor2D_I32 *x2) { 
  nn_assert(x1->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");

  const size_t n = x1->shape[0];
  const size_t m = x1->shape[1];
  const size_t p = x2->shape[1];

  for (size_t i = 0; i < n; i += 1) {
    for (size_t j = 0; j < p; j += 1) {
      int32_t sum = 0;
      for (size_t k = 0; k < m; k += 1) {
        sum += x1->data[i * m + k] * x2->data[k * p + j];
      }
      y->data[i * p + j] = sum + c->data[i * p + j];
    }
  }
}



void nn_linear_i32(Tensor2D_I32 *y, const Tensor2D_I32 *x, const Tensor2D_I32 *weight, const Tensor1D_I32 *bias) { 
  nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];

  for (size_t i = 0; i < batch_size; i += 1) {
    for (size_t j = 0; j < out_features; j += 1) {
      int32_t sum = 0;
      for (size_t k = 0; k < in_features; k += 1) {
        sum += x->data[i * in_features + k] * weight->data[j * in_features + k];
      }
      if (bias) {
        sum += bias->data[j];
      }
      y->data[i * out_features + j] = sum;
    }
  }
}



/* ======================================================================================================== */
/*                                           Non-linear                                                     */
/* ======================================================================================================== */

void nn_relu2d_i32(Tensor2D_I32 *y, const Tensor2D_I32 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  int32_t *x_data = x->data;
  int32_t *y_data = y->data;

  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] > 0 ? x->data[i] : 0;
  }
}


#endif // __NN_Q8_0_H