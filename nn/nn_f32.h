/**
 * @file nn.h
 * @brief The Baremetal-NN Library
 * 
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

#ifndef __NN_F32_H
#define __NN_F32_H

#ifdef CONFIG_BACKEND_RISCV_V
  #include "riscv_vector.h"
#endif

/**
 * Tensor0D_F32
 * 
 * A 0D tensor (scalar) with a float data type.
 */
typedef struct {
  float data;
} Tensor0D_F32;


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
 * Tensor2D_F32
 * 
 * A 2D tensor with a float data type.
 */
typedef struct {
  size_t shape[2];
  float *data;
} Tensor2D_F32;

/**
 * nn_equal_f32
 * 
 * Checks if two floating-point numbers are equal within a relative error.
 * 
 * @param golden The expected value.
 * @param actual The actual value.
 * @param rel_err The relative error tolerance.
 * @return 1 if the numbers are equal within the relative error, 0 otherwise.
 */
static inline uint8_t nn_equal_f32(float golden, float actual, float rel_err) {
  return (fabs(actual - golden) < rel_err) || (fabs((actual - golden) / actual) < rel_err);
}


/* ======================================================================================================== */
/*                                           Tensor Creation                                                */
/* ======================================================================================================== */
/**
 * nn_tensor0d_f32
 * 
 * Creates a 0D tensor with type F32.
 * 
 * @param data The data to store in the tensor.
 */
Tensor0D_F32 *nn_tensor0d_f32(float data) {
  Tensor0D_F32 *tensor = (Tensor0D_F32 *)malloc(sizeof(Tensor0D_F32));
  tensor->data = data;
  return tensor;
}

/**
 * nn_tensor1d_f32
 * 
 * Creates a 1D tensor with type F32.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor1D_F32 *nn_tensor1d_f32(size_t shape[1], const float *data) {
  Tensor1D_F32 *tensor = (Tensor1D_F32 *)malloc(sizeof(Tensor1D_F32));
  tensor->shape[0] = shape[0];

  size_t n_bytes = shape[0] * sizeof(float);
  tensor->data = (float *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
  return tensor;
}

/**
 * nn_tensor2d_f32
 * 
 * Creates a 2D tensor with type F32.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor2D_F32 *nn_tensor2d_f32(size_t shape[2], const float *data) {
  Tensor2D_F32 *tensor = (Tensor2D_F32 *)malloc(sizeof(Tensor2D_F32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];

  size_t n_bytes = shape[0] * shape[1] * sizeof(float);
  tensor->data = (float *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
  return tensor;
}

Tensor0D_F32 *nn_zeros0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(0);
  return tensor;
}

Tensor1D_F32 *nn_zeros1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

Tensor2D_F32 *nn_zeros2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

Tensor0D_F32 *nn_ones0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(1);
  return tensor;
}

Tensor1D_F32 *nn_ones1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

Tensor2D_F32 *nn_ones2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

Tensor0D_F32 *nn_full0d_f32(float data) {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(data);
  return tensor;
}

Tensor1D_F32 *nn_full1d_f32(size_t shape[1], float data) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

Tensor2D_F32 *nn_full2d_f32(size_t shape[2], float data) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

Tensor0D_F32 *nn_rand0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(rand());
  return tensor;
}

Tensor1D_F32 *nn_rand1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
  }
  return tensor;
}

Tensor2D_F32 *nn_rand2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
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
 * nn_print_tensor1d_f32
 * 
 * Prints the content of a 1D tensor with type F32.
 * 
 * @param tensor The 1D tensor to print.
 */
void nn_print_tensor1d_f32(const Tensor1D_F32 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    nn_print_f32(*((float *)tensor->data + i), 3);
    if (i < tensor->shape[0]-1) {
      printf(" ");
    }
  }
  printf("]\n");
}

/**
 * nn_print_tensor2d_f32
 * 
 * Prints the content of a 2D tensor with type F32.
 * 
 * @param tensor The 2D tensor to print.
 */
void nn_print_tensor2d_f32(const Tensor2D_F32 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i != 0) {
      printf(" ");
    }
    printf("[");
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      nn_print_f32(*((float *)tensor->data + i*tensor->shape[1] + j), 3);
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
 * nn_equals0d_f32
 * 
 * Checks if two 0D tensors with type F32 are equal.
 * 
 * @param a The first 0D tensor.
 * @param b The second 0D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals0d_f32(const Tensor0D_F32 *a, const Tensor0D_F32 *b, float rel_err) {
  return nn_equal_f32(a->data, b->data, rel_err);
}

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
uint8_t nn_equals1d_f32(const Tensor1D_F32 *a, const Tensor1D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    if (!nn_equal_f32(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}

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
uint8_t nn_equals2d_f32(const Tensor2D_F32 *a, const Tensor2D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (!nn_equal_f32(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}



/* ======================================================================================================== */
/*                                           Unary                                                          */
/* ======================================================================================================== */
void nn_max1d_f32(Tensor0D_F32 *y, const Tensor1D_F32 *x) {
  size_t n = x->shape[0];
  float *x_data = x->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    vfloat32m1_t vec_max = __riscv_vfmv_s_f_f32m1(-FLT_MAX, 1);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
      vec_max = __riscv_vfredmax_vs_f32m1_f32m1(vec_x, vec_max, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f32m1_f32(vec_max);
  #else  // scalar implementation
    y->data = -FLT_MAX;
    for (size_t i = 0; i < n; i += 1) {
      float val = x->data[i];
      y->data = val > y->data ? val : y->data;
    }
  #endif
}

void nn_max2d_f32(Tensor0D_F32 *y, const Tensor2D_F32 *x) {
  size_t n = x->shape[0] * x->shape[1];
  float *x_data = x->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    vfloat32m1_t vec_max = __riscv_vfmv_s_f_f32m1(-FLT_MAX, 1);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
      vec_max = __riscv_vfredmax_vs_f32m1_f32m1(vec_x, vec_max, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f32m1_f32(vec_max);
  #else  // scalar implementation
    y->data = -FLT_MAX;
    for (size_t i = 0; i < n; i += 1) {
      float val = x->data[i];
      y->data = val > y->data ? val : y->data;
    }
  #endif
}

void nn_min1d_f32(Tensor0D_F32 *y, const Tensor1D_F32 *x) {
  size_t n = x->shape[0];
  float *x_data = x->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(FLT_MAX, 1);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
      vec_min = __riscv_vfredmin_vs_f32m1_f32m1(vec_x, vec_min, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f32m1_f32(vec_min);
  #else  // scalar implementation
    y->data = FLT_MAX;
    for (size_t i = 0; i < n; i += 1) {
      float val = x->data[i];
      y->data = val < y->data ? val : y->data;
    }
  #endif
}

void nn_min2d_f32(Tensor0D_F32 *y, const Tensor2D_F32 *x) {
  size_t n = x->shape[0] * x->shape[1];
  float *x_data = x->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(FLT_MAX, 1);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
      vec_min = __riscv_vfredmin_vs_f32m1_f32m1(vec_x, vec_min, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f32m1_f32(vec_min);
  #else  // scalar implementation
    y->data = FLT_MAX;
    for (size_t i = 0; i < n; i += 1) {
      float val = x->data[i];
      y->data = val < y->data ? val : y->data;
    }
  #endif
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
void nn_add1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float *x1_data = x1->data;
  float *x2_data = x2->data;
  float *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_data, vl);
      vfloat32m1_t vec_x2 = __riscv_vle32_v_f32m1(x2_data, vl);
      vfloat32m1_t vec_y = __riscv_vfadd_vv_f32m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
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
void nn_add2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float *x1_data = x1->data;
  float *x2_data = x2->data;
  float *y_data = y->data;
  
  #ifdef CONFIG_BACKEND_RISCV_V
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_data, vl);
      vfloat32m1_t vec_x2 = __riscv_vle32_v_f32m1(x2_data, vl);
      vfloat32m1_t vec_y = __riscv_vfadd_vv_f32m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
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

void nn_addscalar1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float *x_data = x->data;
  float *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
      vfloat32m1_t vec_y = __riscv_vfadd_vf_f32m1(vec_x, scalar, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = x->data[i] + scalar; 
    }
  #endif
}

void nn_addscalar2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float *x_data = x->data;
  float *y_data = y->data;
  
  #ifdef CONFIG_BACKEND_RISCV_V
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
      vfloat32m1_t vec_y = __riscv_vfadd_vf_f32m1(vec_x, scalar, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = x->data[i] + scalar; 
    }
  #endif
}

/* ======================================================================================================== */
/*                                           Multiplication                                                 */
/* ======================================================================================================== */


void nn_mul1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float *x1_data = x1->data;
  float *x2_data = x2->data;
  float *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_data, vl);
      vfloat32m1_t vec_x2 = __riscv_vle32_v_f32m1(x2_data, vl);
      vfloat32m1_t vec_y = __riscv_vfmul_vv_f32m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = x1->data[i] * x2->data[i]; 
    }
  #endif
}

void nn_mul2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float *x1_data = x1->data;
  float *x2_data = x2->data;
  float *y_data = y->data;
  
  #ifdef CONFIG_BACKEND_RISCV_V
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_data, vl);
      vfloat32m1_t vec_x2 = __riscv_vle32_v_f32m1(x2_data, vl);
      vfloat32m1_t vec_y = __riscv_vfmul_vv_f32m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = x1->data[i] * x2->data[i]; 
    }
  #endif
}

void nn_mulscalar1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float *x_data = x->data;  
  float *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
      vfloat32m1_t vec_y = __riscv_vfmul_vf_f32m1(vec_x, scalar, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = x->data[i] * scalar; 
    }
  #endif
}

void nn_mulscalar2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float *x_data = x->data;
  float *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
      vfloat32m1_t vec_y = __riscv_vfmul_vf_f32m1(vec_x, scalar, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = x->data[i] * scalar; 
    }
  #endif
}


/* ======================================================================================================== */
/*                                           MatMul                                                         */
/* ======================================================================================================== */
void nn_dot_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot dot tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot dot tensors of different shapes");

  size_t n = y->shape[0];
  float *x1_data = x1->data;
  float *x2_data = x2->data;
  float *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_data, vl);
      vfloat32m1_t vec_x2 = __riscv_vle32_v_f32m1(x2_data, vl);
      vfloat32m1_t vec_y = __riscv_vfmul_vv_f32m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    float sum = 0.0;
    for (size_t i = 0; i < n; i += 1) {
      sum += x1->data[i] * x2->data[i];
    }
    y->data[0] = sum;
  #endif
}

void nn_mm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) { 
  nn_assert(x1->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");

  const size_t batch_size = x1->shape[0];
  const size_t in_features = x1->shape[1];
  const size_t out_features = x2->shape[0];
  
  float *x1_batch_data = x1->data;
  float *x2_batch_data = x2->data;
  float *y_batch_data = y->data;

  for (size_t i = 0; i < batch_size; i += 1) {
    #ifdef CONFIG_BACKEND_RISCV_V
      float *x1_data = x1_batch_data;
      float *x2_data = x2_batch_data;
      float *y_data = y_batch_data;

      size_t vlmax = __riscv_vsetvlmax_e32m1();
      for (size_t j = 0; j < out_features; j += 1) {
        vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
        vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
        
        size_t n = in_features;
        
        while (n > 0) {
          size_t vl = __riscv_vsetvl_e32m1(n);
          vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x1_data, vl);
          vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(x2_data, vl);
          vec_sum = __riscv_vfmacc_vv_f32m1(vec_sum, vec_x, vec_y, vl);
          
          x1_data += vl;
          x2_data += vl;
          n -= vl;
        }
        #ifdef CONFIG_DEBUG_RISCV_V_USE_REDOSUM
          vec_sum = __riscv_vfredosum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #else
          vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #endif
        y_data[j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        
        x1_data -= in_features;
      }
      
      x1_batch_data += in_features;
      y_batch_data += out_features;
    #else  // scalar implementation
      for (size_t j = 0; j < out_features; j += 1) {
        float sum = 0.f;
        for (size_t k = 0; k < in_features; k += 1) {
          sum += x1->data[i * in_features + k] * x2->data[j * in_features + k];
        }
        y->data[i * out_features + j] = sum;
      }
    #endif
  }
}


void nn_addmm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const Tensor2D_F32 *weight, const Tensor1D_F32 *bias) { 
  nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  nn_assert(bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];
  
  float *x_batch_data = x->data;
  float *y_batch_data = y->data;

  for (size_t i = 0; i < batch_size; i += 1) {
    #ifdef CONFIG_BACKEND_RISCV_V
      float *weight_data = weight->data;
      float *bias_data = bias->data;
      float *x_data = x_batch_data;
      float *y_data = y_batch_data;

      size_t vlmax = __riscv_vsetvlmax_e32m1();

      for (size_t j = 0; j < out_features; j += 1) {
        vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
        vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
        
        size_t n = in_features;
        
        while (n > 0) {
          size_t vl = __riscv_vsetvl_e32m1(n);
          vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
          vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(weight_data, vl);
          vec_sum = __riscv_vfmacc_vv_f32m1(vec_sum, vec_x, vec_y, vl);
          
          x_data += vl;
          weight_data += vl;
          n -= vl;
        }
        
        #ifdef CONFIG_DEBUG_RISCV_V_USE_REDOSUM
          vec_sum = __riscv_vfredosum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #else
          vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #endif
        y_data[j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum) + bias_data[j];
        x_data -= in_features;
      }
      
      x_batch_data += in_features;
      y_batch_data += out_features;
    #else  // scalar implementation
      for (size_t j = 0; j < out_features; j += 1) {
        float sum = 0.f;
        for (size_t k = 0; k < in_features; k += 1) {
          sum += x->data[i * in_features + k] * weight->data[j * in_features + k];
        }
        y->data[i * out_features + j] = sum + bias->data[j];
      }
    #endif
  }
}



/* ======================================================================================================== */
/*                                           Non-linear                                                     */
/* ======================================================================================================== */
void nn_elu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float alpha) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ELU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (x->data[i] > 0) {
      y->data[i] = x->data[i];
    }
    else {
      y->data[i] = alpha * (expf(x->data[i]) - 1.f);
    }
  }
}

void nn_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float *x_data = x->data;
  float *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_V
    float zero = 0.0f;

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n);
      vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
      vfloat32m1_t vec_y = __riscv_vfmax_vf_f32m1(vec_x, zero, vl);
      __riscv_vse32_v_f32m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      float x_val = x->data[i];
      y->data[i] = x_val > 0 ? x_val : 0;
    }
  #endif
}


void nn_softmax1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float sum = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    sum += expf(x->data[i]);
  }

  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = expf(x->data[i]) / sum;
  }
}

void nn_softmax2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, size_t dim) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  if (dim == 0) {
    for (size_t i = 0; i < y->shape[0]; i += 1) {
      size_t n = y->shape[1];
      float sum = 0.0f;
      for (size_t j = 0; j < n; j += 1) {
        sum += expf(x->data[i * n + j]);
      }

      for (size_t j = 0; j < n; j += 1) {
        y->data[i * n + j] = expf(x->data[i * n + j]) / sum;
      }
    }
  }
  else if (dim == 1) {
    // HACK: fix batch size
    size_t n = y->shape[1];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i += 1) {
      sum += expf(x->data[i]);
    }

    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = expf(x->data[i]) / sum;
    }
  }
  else {
    nn_assert(0, "Invalid dimension for softmax");
  }
}



void nn_tanh2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x->data[i];
    y->data[i] = tanh(x_val);
  }
}




#endif // __NN_F32_H