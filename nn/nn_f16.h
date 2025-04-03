/**
 * @file nn_f16.h
 * @brief Baremetal-NN Library functions for half-precision floating-point (fp16) numbers
 *
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

#ifndef __NN_F16_H
#define __NN_F16_H

#include "float16.h"


#ifdef CONFIG_BACKEND_RISCV_V
  #include "riscv_vector.h"
#endif

/**
 * Tensor0D_F16
 *
 * @brief A 0D tensor (scalar) with a half-precision floating-point data type.
 */
typedef struct {
  float16_t data;
} Tensor0D_F16;


/**
 * Tensor1D_F16
 *
 * @brief A 1D tensor with a half-precision floating-point data type.
 */
typedef struct {
  size_t shape[1];
  float16_t *data;
} Tensor1D_F16;


/**
 * Tensor2D_F16
 *
 * @brief A 2D tensor with a half-precision floating-point data type.
 */
typedef struct {
  size_t shape[2];
  float16_t *data;
} Tensor2D_F16;

/**
 * nn_equal_f16
 *
 * @brief Checks if two half-precision floating-point numbers are equal within a relative error.
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
 * @brief Creates a 0D tensor with type F16.
 *
 * @param data The data to store in the tensor.
 * @return The created tensor.
 */
Tensor0D_F16 *nn_tensor0d_f16(float16_t data) {
  Tensor0D_F16 *tensor = (Tensor0D_F16 *)malloc(sizeof(Tensor0D_F16));
  tensor->data = data;
  return tensor;
}

/**
 * nn_tensor1d_f16
 *
 * @brief Creates a 1D tensor with type F16.
 *
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 * @return The created tensor.
 */
Tensor1D_F16 *nn_tensor1d_f16(size_t shape[1], const float16_t *data) {
  Tensor1D_F16 *tensor = (Tensor1D_F16 *)malloc(sizeof(Tensor1D_F16));
  tensor->shape[0] = shape[0];

  size_t n_bytes = shape[0] * sizeof(float16_t);
  tensor->data = (float16_t *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
  return tensor;
}

/**
 * nn_tensor2d_f16
 *
 * @brief Creates a 2D tensor with type F16.
 *
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 * @return The created tensor.
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
  return tensor;
}

/**
 * nn_zeros0d_f16
 *
 * @brief Creates a 0D tensor with type F16 and initializes it to 0.
 *
 * @return The created tensor.
 */
Tensor0D_F16 *nn_zeros0d_f16() {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(0);
  return tensor;
}

/**
 * nn_zeros1d_f16
 *
 * @brief Creates a 1D tensor with type F16 and initializes it to 0.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor1D_F16 *nn_zeros1d_f16(size_t shape[1]) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

/**
 * nn_zeros2d_f16
 *
 * @brief Creates a 2D tensor with type F16 and initializes it to 0.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor2D_F16 *nn_zeros2d_f16(size_t shape[2]) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

/**
 * nn_ones0d_f16
 *
 * @brief Creates a 0D tensor with type F16 and initializes it to 1.
 *
 * @return The created tensor.
 */
Tensor0D_F16 *nn_ones0d_f16() {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(1);
  return tensor;
}

/**
 * nn_ones1d_f16
 *
 * @brief Creates a 1D tensor with type F16 and initializes it to 1.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor1D_F16 *nn_ones1d_f16(size_t shape[1]) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

/**
 * nn_ones2d_f16
 *
 * @brief Creates a 2D tensor with type F16 and initializes it to 1.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor2D_F16 *nn_ones2d_f16(size_t shape[2]) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

/**
 * nn_full0d_f16
 *
 * @brief Creates a 0D tensor with type F16 and initializes it to a specific value.
 *
 * @param data The value to initialize the tensor to.
 * @return The created tensor.
 */
Tensor0D_F16 *nn_full0d_f16(float16_t data) {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(data);
  return tensor;
}

/**
 * nn_full1d_f16
 *
 * @brief Creates a 1D tensor with type F16 and initializes it to a specific value.
 *
 * @param shape The shape of the tensor.
 * @param data The value to initialize the tensor to.
 * @return The created tensor.
 */
Tensor1D_F16 *nn_full1d_f16(size_t shape[1], float16_t data) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

/**
 * nn_full2d_f16
 *
 * @brief Creates a 2D tensor with type F16 and initializes it to a specific value.
 *
 * @param shape The shape of the tensor.
 * @param data The value to initialize the tensor to.
 * @return The created tensor.
 */
Tensor2D_F16 *nn_full2d_f16(size_t shape[2], float16_t data) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

/**
 * nn_rand0d_f16
 *
 * @brief Creates a 0D tensor with type F16 and initializes it to a random value.
 *
 * @return The created tensor.
 */
Tensor0D_F16 *nn_rand0d_f16() {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(as_f16(rand()));
  return tensor;
}

/**
 * nn_rand1d_f16
 *
 * @brief Creates a 1D tensor with type F16 and initializes it to a random value.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor1D_F16 *nn_rand1d_f16(size_t shape[1]) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = as_f16(rand());
  }
  return tensor;
}

/**
 * nn_rand2d_f16
 *
 * @brief Creates a 2D tensor with type F16 and initializes it to a random value.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
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
 * @brief Prints a half-precision floating-point number.
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
 * @brief Prints the content of a 1D tensor with type F16.
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
 * @brief Prints the content of a 2D tensor with type F16.
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
 * @brief Checks if two 0D tensors with type F16 are equal.
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
 * @brief Checks if two 1D tensors with type F16 are equal.
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
 * @brief Checks if two 2D tensors with type F16 are equal.
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
/**
 * nn_max1d_f16
 *
 * @brief Finds the maximum value in a 1D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_max1d_f16(Tensor0D_F16 *y, const Tensor1D_F16 *x) {
  size_t n = x->shape[0];
  float16_t *x_data = x->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    vfloat16m1_t vec_max = __riscv_vfmv_v_f_f16m1(-FLT16_MAX, 1);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vec_max = __riscv_vfredmax_vs_f16m1_f16m1(vec_x, vec_max, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f16m1_f16(vec_max);
  #else  // scalar implementation
    y->data = -FLT16_MAX;
    for (size_t i = 0; i < n; i += 1) {
      float val = as_f32(x->data[i]);
      y->data = val > y->data ? val : y->data;
    }
  #endif
}

void nn_max2d_f16(Tensor0D_F16 *y, const Tensor2D_F16 *x) {
  size_t n = x->shape[0] * x->shape[1];
  float16_t *x_data = x->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    vfloat16m1_t vec_max = __riscv_vfmv_v_f_f16m1(-FLT16_MAX, 1);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vec_max = __riscv_vfredmax_vs_f16m1_f16m1(vec_x, vec_max, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f16m1_f16(vec_max);
  #else  // scalar implementation
    y->data = -FLT16_MAX;
    for (size_t i = 0; i < n; i += 1) {
      float val = as_f32(x->data[i]);
      y->data = val > y->data ? val : y->data;
    }
  #endif
}


/**
 * nn_min1d_f16
 *
 * @brief Finds the minimum value in a 1D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_min1d_f16(Tensor0D_F16 *y, const Tensor1D_F16 *x) {
  size_t n = x->shape[0];
  float16_t *x_data = x->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    vfloat16m1_t vec_min = __riscv_vfmv_v_f_f16m1(FLT16_MAX, 1);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vec_min = __riscv_vfredmin_vs_f16m1_f16m1(vec_x, vec_min, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f16m1_f16(vec_min);
  #else  // scalar implementation
    y->data = FLT16_MAX;
    for (size_t i = 0; i < n; i += 1) {
      float val = as_f32(x->data[i]);
      y->data = val < y->data ? val : y->data;
    }
  #endif
}


/**
 * nn_min2d_f16
 *
 * @brief Finds the minimum value in a 2D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_min2d_f16(Tensor0D_F16 *y, const Tensor2D_F16 *x) {
  size_t n = x->shape[0] * x->shape[1];
  float16_t *x_data = x->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    vfloat16m1_t vec_min = __riscv_vfmv_v_f_f16m1(FLT16_MAX, 1);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vec_min = __riscv_vfredmin_vs_f16m1_f16m1(vec_x, vec_min, vl);
      x_data += vl;
      n -= vl;
    }
    y->data = __riscv_vfmv_f_s_f16m1_f16(vec_min);
  #else  // scalar implementation
    y->data = FLT16_MAX;
    for (size_t i = 0; i < n; i += 1) {
      float val = as_f32(x->data[i]);
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
 * @brief Adds x1 and x2 element-wise and stores the result in y.
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
  float16_t *x1_data = x1->data;
  float16_t *x2_data = x2->data;
  float16_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x1 = __riscv_vle16_v_f16m1(x1_data, vl);
      vfloat16m1_t vec_x2 = __riscv_vle16_v_f16m1(x2_data, vl);
      vfloat16m1_t vec_y = __riscv_vfadd_vv_f16m1(vec_x1, vec_x2, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = as_f16(as_f32(x1->data[i]) + as_f32(x2->data[i]));
    }
  #endif
}



/**
 * nn_add2d_f32
 *
 * @brief Adds x1 and x2 element-wise and stores the result in y.
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
  float16_t *x1_data = x1->data;
  float16_t *x2_data = x2->data;
  float16_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x1 = __riscv_vle16_v_f16m1(x1_data, vl);
      vfloat16m1_t vec_x2 = __riscv_vle16_v_f16m1(x2_data, vl);
      vfloat16m1_t vec_y = __riscv_vfadd_vv_f16m1(vec_x1, vec_x2, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = as_f16(as_f32(x1->data[i]) + as_f32(x2->data[i]));
    }
  #endif
}

/**
 * nn_addscalar1d_f16
 *
 * @brief Adds a scalar to a 1D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param scalar The scalar to add.
 */
void nn_addscalar1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float16_t *x_data = x->data;
  float16_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vfloat16m1_t vec_y = __riscv_vfadd_vf_f16m1(vec_x, scalar, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = as_f16(as_f32(x->data[i]) + as_f32(scalar));
    }
  #endif
}

/**
 * nn_addscalar2d_f16
 *
 * @brief Adds a scalar to a 2D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param scalar The scalar to add.
 */
void nn_addscalar2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float16_t *x_data = x->data;
  float16_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vfloat16m1_t vec_y = __riscv_vfadd_vf_f16m1(vec_x, scalar, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = as_f16(as_f32(x->data[i]) + as_f32(scalar));
    }
  #endif
}




/* ======================================================================================================== */
/*                                           Multiplication                                                 */
/* ======================================================================================================== */

/**
 * nn_mul1d_f16
 *
 * @brief Multiplies x1 and x2 element-wise and stores the result in y.
 *
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor.
 */
void nn_mul1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x1, const Tensor1D_F16 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float16_t *x1_data = x1->data;
  float16_t *x2_data = x2->data;
  float16_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x1 = __riscv_vle16_v_f16m1(x1_data, vl);
      vfloat16m1_t vec_x2 = __riscv_vle16_v_f16m1(x2_data, vl);
      vfloat16m1_t vec_y = __riscv_vfadd_vv_f16m1(vec_x1, vec_x2, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = as_f16(as_f32(x1->data[i]) * as_f32(x2->data[i]));
    }
  #endif
}

/**
 * nn_mul2d_f16
 *
 * @brief Multiplies x1 and x2 element-wise and stores the result in y.
 *
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor.
 */
void nn_mul2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float16_t *x1_data = x1->data;
  float16_t *x2_data = x2->data;
  float16_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x1 = __riscv_vle16_v_f16m1(x1_data, vl);
      vfloat16m1_t vec_x2 = __riscv_vle16_v_f16m1(x2_data, vl);
      vfloat16m1_t vec_y = __riscv_vfadd_vv_f16m1(vec_x1, vec_x2, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = as_f16(as_f32(x1->data[i]) * as_f32(x2->data[i]));
    }
  #endif
}

/**
 * nn_mulscalar1d_f16
 *
 * @brief Multiplies a scalar with a 1D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param scalar The scalar to multiply.
 */
void nn_mulscalar1d_f16(Tensor1D_F16 *y, const Tensor1D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float16_t *x_data = x->data;
  float16_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vfloat16m1_t vec_y = __riscv_vfadd_vf_f16m1(vec_x, scalar, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = as_f16(as_f32(x->data[i]) * as_f32(scalar));
    }
  #endif
}

/**
 * nn_mulscalar2d_f16
 *
 * @brief Multiplies a scalar with a 2D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param scalar The scalar to multiply.
 */
void nn_mulscalar2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, float16_t scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float16_t *x_data = x->data;
  float16_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vfloat16m1_t vec_y = __riscv_vfadd_vf_f16m1(vec_x, scalar, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = as_f16(as_f32(x->data[i]) * as_f32(scalar));
    }
  #endif
}



/* ======================================================================================================== */
/*                                           MatMul                                                         */
/* ======================================================================================================== */
/**
 * nn_dot_f16
 *
 * @brief Performs a dot product of two 1D tensors with type F16.
 *
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor.
 */
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

/**
 * nn_mm_f16
 *
 * @brief Performs a matrix multiplication of the matrices x1 and x2.
 *
 * @param y The output tensor, shape (n, p)
 * @param x1 The first input tensor, shape (n, m)
 * @param x2 The second input tensor, shape (m, p)
 */
void nn_mm_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) {
  nn_assert(x1->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");

  const size_t n = x1->shape[0];
  const size_t m = x1->shape[1];
  const size_t p = x2->shape[1];

  for (size_t i = 0; i < n; i += 1) {
    #ifdef CONFIG_BACKEND_RISCV_ZVFH
      float16_t *x1_row = x1->data + i * m;
      float16_t *y_row = y->data + i * p;

      size_t vlmax = __riscv_vsetvlmax_e16m1();
      for (size_t j = 0; j < p; j += 1) {
        vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
        vfloat16m1_t vec_sum = __riscv_vfmv_v_f_f16m1(0, vlmax);

        float16_t *x1_ptr = x1_row;
        float16_t *x2_ptr = x2->data + j;
        size_t k = m;

        while (k > 0) {
          size_t vl = __riscv_vsetvl_e16m1(k);
          vfloat16m1_t vec_x1 = __riscv_vle16_v_f16m1(x1_ptr, vl);
          vfloat16m1_t vec_x2 = __riscv_vlse16_v_f16m1(x2_ptr, p * sizeof(float16_t), vl);
          vec_sum = __riscv_vfmacc_vv_f16m1(vec_sum, vec_x1, vec_x2, vl);

          x1_ptr += vl;
          x2_ptr += vl * p;
          k -= vl;
        }

        #ifdef CONFIG_DEBUG_RISCV_V_USE_REDOSUM
          vec_sum = __riscv_vfredosum_vs_f16m1_f16m1(vec_sum, vec_zero, vlmax);
        #else
          vec_sum = __riscv_vfredusum_vs_f16m1_f16m1(vec_sum, vec_zero, vlmax);
        #endif
        y_row[j] = __riscv_vfmv_f_s_f16m1_f16(vec_sum);
      }
    #else
      for (size_t j = 0; j < p; j += 1) {
        float sum = 0.f;
        for (size_t k = 0; k < m; k += 1) {
          sum += as_f32(x1->data[i * m + k]) * as_f32(x2->data[k * p + j]);
        }
        y->data[i * p + j] = as_f16(sum);
      }
    #endif
  }
}

/**
 * nn_addmm_f16
 *
 * @brief Performs a matrix multiplication of the matrices x1 and x2.
 *
 * @param y The output tensor, shape (n, p)
 * @param c The bias tensor, shape (n, p)
 * @param x1 The first input tensor, shape (n, m)
 * @param x2 The second input tensor, shape (m, p)
 */
void nn_addmm_f16(Tensor2D_F16 *y, const Tensor2D_F16 *c, const Tensor2D_F16 *x1, const Tensor2D_F16 *x2) {
  nn_assert(x1->shape[1] == x2->shape[0], "Cannot perform Linear on tensors of different shapes");
  nn_assert(y->shape[0] == c->shape[0] && y->shape[1] == x2->shape[1], "Cannot perform Linear on tensors of different shapes");

  const size_t n = x1->shape[0];
  const size_t m = x1->shape[1];
  const size_t p = x2->shape[1];

  for (size_t i = 0; i < n; i += 1) {
    #ifdef CONFIG_BACKEND_RISCV_ZVFH
      float16_t *x1_row = x1->data + i * m;
      float16_t *y_row = y->data + i * p;

      size_t vlmax = __riscv_vsetvlmax_e16m1();
      for (size_t j = 0; j < p; j += 1) {
        vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
        vfloat16m1_t vec_sum = __riscv_vfmv_v_f_f16m1(0, vlmax);

        float16_t *x1_ptr = x1_row;
        float16_t *x2_ptr = x2->data + j;
        size_t k = m;

        while (k > 0) {
          size_t vl = __riscv_vsetvl_e16m1(k);
          vfloat16m1_t vec_x1 = __riscv_vle16_v_f16m1(x1_ptr, vl);
          vfloat16m1_t vec_x2 = __riscv_vlse16_v_f16m1(x2_ptr, p * sizeof(float16_t), vl);
          vec_sum = __riscv_vfmacc_vv_f16m1(vec_sum, vec_x1, vec_x2, vl);

          x1_ptr += vl;
          x2_ptr += vl * p;
          k -= vl;
        }

        #ifdef CONFIG_DEBUG_RISCV_V_USE_REDOSUM
          vec_sum = __riscv_vfredosum_vs_f16m1_f16m1(vec_sum, vec_zero, vlmax);
        #else
          vec_sum = __riscv_vfredusum_vs_f16m1_f16m1(vec_sum, vec_zero, vlmax);
        #endif
        y_row[j] = __riscv_vfmv_f_s_f16m1_f16(vec_sum) + c->data[i * p + j];
      }

      x1_row += m;
      y_row += p;
    #else
      for (size_t j = 0; j < p; j += 1) {
        float sum = 0.f;
        for (size_t k = 0; k < m; k += 1) {
          sum += as_f32(x1->data[i * m + k]) * as_f32(x2->data[k * p + j]);
        }
        y->data[i * p + j] = as_f16(sum + as_f32(c->data[i * p + j]));
      }
    #endif
  }
}

/**
 * nn_linear_f16
 *
 * @brief Linear neural network layer.
 *
 * y[i][j] = x[i][k] * weight[j][k] + bias[j]
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param weight The weight tensor.
 * @param bias The bias tensor.
 */
void nn_linear_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x, const Tensor2D_F16 *weight, const Tensor1D_F16 *bias) {
  nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];

  float16_t *x_batch_data = x->data;
  float16_t *y_batch_data = y->data;

  for (size_t i = 0; i < batch_size; i += 1) {
    #ifdef CONFIG_BACKEND_RISCV_ZVFH
      float16_t *x_data = x_batch_data;
      float16_t *y_data = y_batch_data;

      size_t vlmax = __riscv_vsetvlmax_e16m1();

      for (size_t j = 0; j < out_features; j += 1) {
        vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
        vfloat16m1_t vec_sum = __riscv_vfmv_v_f_f16m1(0, vlmax);

        float16_t *weight_row = weight->data + j * in_features;
        size_t n = in_features;

        while (n > 0) {
          size_t vl = __riscv_vsetvl_e16m1(n);
          vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
          vfloat16m1_t vec_w = __riscv_vle16_v_f16m1(weight_row, vl);
          vec_sum = __riscv_vfmacc_vv_f16m1(vec_sum, vec_x, vec_w, vl);

          x_data += vl;
          weight_row += vl;
          n -= vl;
        }

        #ifdef CONFIG_DEBUG_RISCV_V_USE_REDOSUM
          vec_sum = __riscv_vfredosum_vs_f16m1_f16m1(vec_sum, vec_zero, vlmax);
        #else
          vec_sum = __riscv_vfredusum_vs_f16m1_f16m1(vec_sum, vec_zero, vlmax);
        #endif

        float16_t sum = __riscv_vfmv_f_s_f16m1_f16(vec_sum);
        if (bias) {
          sum = as_f16(as_f32(sum) + as_f32(bias->data[j]));
        }
        y_data[j] = sum;
        x_data = x_batch_data; // reset x_data pointer for next output feature
      }

      x_batch_data += in_features;
      y_batch_data += out_features;
    #else  // scalar implementation
      for (size_t j = 0; j < out_features; j += 1) {
        float sum = 0.f;
        for (size_t k = 0; k < in_features; k += 1) {
          sum += as_f32(x->data[i * in_features + k]) * as_f32(weight->data[j * in_features + k]);
        }
        if (bias) {
          sum += as_f32(bias->data[j]);
        }
        y->data[i * out_features + j] = as_f16(sum);
      }
    #endif
  }
}



/* ======================================================================================================== */
/*                                           Non-linear                                                     */
/* ======================================================================================================== */

/**
 * nn_elu2d_f16
 *
 * @brief Applies the ELU activation function to a 2D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param alpha The alpha parameter for the ELU activation function.
 */
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

/**
 * nn_relu2d_f16
 *
 * @brief Applies the ReLU activation function to a 2D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_relu2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  float16_t *x_data = x->data;
  float16_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_ZVFH
    float16_t zero = 0.0f;

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e16m1(n);
      vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x_data, vl);
      vfloat16m1_t vec_y = __riscv_vfmax_vf_f16m1(vec_x, zero, vl);
      __riscv_vse16_v_f16m1(y_data, vec_y, vl);
      x_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      float x_val = as_f32(x->data[i]);
      y->data[i] = x_val > 0 ? as_f16(x_val) : 0;
    }
  #endif
}

/**
 * nn_softmax1d_f16
 *
 * @brief Applies the softmax activation function to a 1D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 */
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

/**
 * nn_tanh2d_f16
 *
 * @brief Applies the tanh activation function to a 2D tensor with type F16.
 *
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_tanh2d_f16(Tensor2D_F16 *y, const Tensor2D_F16 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = as_f32(x->data[i]);
    y->data[i] = as_f16(tanh(x_val));
  }
}



#endif // __NN_F32_H