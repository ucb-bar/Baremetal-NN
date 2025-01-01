/**
 * @file nn.h
 * @brief The Baremetal-NN Library
 * 
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

#ifndef __NN_F32_H
#define __NN_F32_H

#include <float.h>

#ifdef CONFIG_BACKEND_RISCV_V
  #include "riscv_vector.h"
#endif

/**
 * A 0D tensor (scalar) with a float data type.
 */
typedef struct {
  float data;
} Tensor0D_F32;


/**
 * A 1D tensor with a float data type.
 */
typedef struct {
  size_t shape[1];
  float *data;
} Tensor1D_F32;


/**
 * A 2D tensor with a float data type.
 */
typedef struct {
  size_t shape[2];
  float *data;
} Tensor2D_F32;

/**
 * A 3D tensor with a float data type.
 */
typedef struct {
  size_t shape[3];
  float *data;
} Tensor3D_F32;

/**
 * A 4D tensor with a float data type.
 */
typedef struct {
  size_t shape[4];
  float *data;
} Tensor4D_F32;


/**
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
 * Creates a 0D floating-point data tensor.
 * 
 * This method always allocates memory for the tensor. If `data` is specified, it copies the data from `data` to the tensor.
 * 
 * @param data If specified, the data to copy from and store in the tensor.
 */
Tensor0D_F32 *nn_tensor0d_f32(float data) {
  Tensor0D_F32 *tensor = (Tensor0D_F32 *)malloc(sizeof(Tensor0D_F32));
  tensor->data = data;
  return tensor;
}

/**
 * Creates a 1D floating-point data tensor, with the shape defined by the 1-element array `shape`.
 * 
 * This method always allocates memory for the tensor. If `data` is specified, it copies the data from `data` to the tensor.
 * 
 * @param shape The shape of the tensor.
 * @param data If specified, the data to copy from and store in the tensor.
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
 * Creates a 2D floating-point data tensor, with the shape defined by the 2-element array `shape`.
 * 
 * This method always allocates memory for the tensor. If `data` is specified, it copies the data from `data` to the tensor.
 * 
 * @param shape The shape of the tensor.
 * @param data If specified, the data to copy from and store in the tensor.
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

/**
 * Creates a 3D floating-point data tensor, with the shape defined by the 3-element array `shape`.
 * 
 * This method always allocates memory for the tensor. If `data` is specified, it copies the data from `data` to the tensor.
 * 
 * @param shape The shape of the tensor.
 * @param data If specified, the data to copy from and store in the tensor.
 */
Tensor3D_F32 *nn_tensor3d_f32(size_t shape[3], const float *data) {
  Tensor3D_F32 *tensor = (Tensor3D_F32 *)malloc(sizeof(Tensor3D_F32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];
  tensor->shape[2] = shape[2];

  size_t n_bytes = shape[0] * shape[1] * shape[2] * sizeof(float);
  tensor->data = (float *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
  return tensor;
}

/**
 * Creates a 4D floating-point data tensor, with the shape defined by the 4-element array `shape`.
 * 
 * This method always allocates memory for the tensor. If `data` is specified, it copies the data from `data` to the tensor.
 * 
 * @param shape The shape of the tensor.
 * @param data If specified, the data to copy from and store in the tensor.
 */
Tensor4D_F32 *nn_tensor4d_f32(size_t shape[4], const float *data) {
  Tensor4D_F32 *tensor = (Tensor4D_F32 *)malloc(sizeof(Tensor4D_F32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];
  tensor->shape[2] = shape[2];
  tensor->shape[3] = shape[3];

  size_t n_bytes = shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float);
  tensor->data = (float *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
  return tensor;
}

/**
 * Converts data into a tensor, with the shape defined by the 1-element array `shape`.
 * 
 * @param shape The shape of the tensor.
 * @param data The pointer to the tensor data.
 */
Tensor1D_F32 *nn_as_tensor1d_f32(size_t shape[1], float *data) {
  Tensor1D_F32 *tensor = (Tensor1D_F32 *)malloc(sizeof(Tensor1D_F32));
  tensor->shape[0] = shape[0];
  tensor->data = data;
  return tensor;
}

/**
 * Converts data into a tensor, with the shape defined by the 2-element array `shape`.
 * 
 * @param shape The shape of the tensor.
 * @param data The pointer to the tensor data.
 */
Tensor2D_F32 *nn_as_tensor2d_f32(size_t shape[2], float *data) {
  Tensor2D_F32 *tensor = (Tensor2D_F32 *)malloc(sizeof(Tensor2D_F32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];
  tensor->data = data;
  return tensor;
}

/**
 * Converts data into a tensor, with the shape defined by the 3-element array `shape`.
 * 
 * @param shape The shape of the tensor.
 * @param data The pointer to the tensor data.
 */
Tensor3D_F32 *nn_as_tensor3d_f32(size_t shape[3], float *data) {
  Tensor3D_F32 *tensor = (Tensor3D_F32 *)malloc(sizeof(Tensor3D_F32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];
  tensor->shape[2] = shape[2];
  tensor->data = data;
  return tensor;
}

/**
 * Converts data into a tensor, with the shape defined by the 4-element array `shape`.
 * 
 * @param shape The shape of the tensor.
 * @param data The pointer to the tensor data.
 */
Tensor4D_F32 *nn_as_tensor4d_f32(size_t shape[4], float *data) {
  Tensor4D_F32 *tensor = (Tensor4D_F32 *)malloc(sizeof(Tensor4D_F32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];
  tensor->shape[2] = shape[2];
  tensor->shape[3] = shape[3];
  tensor->data = data;
  return tensor;
}


/**
 * Returns a 0D floating-point tensor (scalar) filled with the scalar value 0.
 */
Tensor0D_F32 *nn_zeros0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(0);
  return tensor;
}

/**
 * Returns a 1D floating-point tensor filled with the scalar value 0, with the shape defined by the 1-element array `shape`.
 */
Tensor1D_F32 *nn_zeros1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

/**
 * Returns a 2D floating-point data tensor filled with the scalar value 0, with the shape defined by the 2-element array `shape`.
 */
Tensor2D_F32 *nn_zeros2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

/**
 * Returns a 3D floating-point data tensor filled with the scalar value 0, with the shape defined by the 3-element array `shape`. 
 */
Tensor3D_F32 *nn_zeros3d_f32(size_t shape[3]) {
  Tensor3D_F32 *tensor = nn_tensor3d_f32(shape, NULL);
  size_t n = shape[0] * shape[1] * shape[2];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

/**
 * Returns a 4D floating-point data tensor filled with the scalar value 0, with the shape defined by the 4-element array `shape`. 
 */
Tensor4D_F32 *nn_zeros4d_f32(size_t shape[4]) {
  Tensor4D_F32 *tensor = nn_tensor4d_f32(shape, NULL);
  size_t n = shape[0] * shape[1] * shape[2] * shape[3];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

/**
 * Returns a 0D floating-point data tensor (scalar) filled with the scalar value 1.
 */
Tensor0D_F32 *nn_ones0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(1);
  return tensor;
}

/**
 * Returns a 1D floating-point data tensor filled with the scalar value 1, with the shape defined by the 1-element array `shape`.
 */
Tensor1D_F32 *nn_ones1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

/**
 * Returns a 2D floating-point data tensor filled with the scalar value 1, with the shape defined by the 2-element array `shape`.
 */
Tensor2D_F32 *nn_ones2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

/**
 * Returns a 0D floating-point data tensor (scalar) filled with the scalar value `data`.
 */
Tensor0D_F32 *nn_full0d_f32(float data) {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(data);
  return tensor;
}

/**
 * Returns a 1D floating-point data tensor filled with the scalar value `data`, with the shape defined by the 1-element array `shape`.
 */
Tensor1D_F32 *nn_full1d_f32(size_t shape[1], float data) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

/**
 * Returns a 2D floating-point data tensor filled with the scalar value `data`, with the shape defined by the 2-element array `shape`.
 */
Tensor2D_F32 *nn_full2d_f32(size_t shape[2], float data) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

/**
 * Returns a 0D floating-point data tensor (scalar) filled with a random floating-point number.
 */
Tensor0D_F32 *nn_rand0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(rand());
  return tensor;
}

/**
 * Returns a 1D floating-point data tensor filled with random floating-point numbers, with the shape defined by the 1-element array `shape`.
 */
Tensor1D_F32 *nn_rand1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
  }
  return tensor;
}

/**
 * Returns a 2D floating-point data tensor filled with random floating-point numbers, with the shape defined by the 2-element array `shape`.
 */
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
 * Prints the content of a 1D floating-point data tensor.
 * 
 * @param tensor The 1D floating-point data tensor to print.
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
 * Prints the content of a 2D floating-point data tensor.
 * 
 * @param tensor The 2D floating-point data tensor to print.
 */
void nn_print_tensor2d_f32(const Tensor2D_F32 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i == 0) {
      printf("[");
    }
    else {
      printf(" [");
    }
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      nn_print_f32(*((float *)tensor->data + i*tensor->shape[1] + j), 3);
      if (j < tensor->shape[1]-1) {
        printf(" ");
      }
    }
    printf(" ]");
    if (i < tensor->shape[0]-1) {
      printf("\n");
    }
  }
  printf("]\n");
}

/**
 * Prints the content of a 3D floating-point data tensor.
 * 
 * @param tensor The 3D floating-point data tensor to print.
 */
void nn_print_tensor3d_f32(const Tensor3D_F32 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i == 0) {
      printf("[");
    }
    else {
      printf("\n [");
    }
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      if (j == 0) {
        printf("[");
      }
      else {
        printf("  [");
      }
      for (size_t k=0; k<tensor->shape[2]; k+=1) {
        nn_print_f32(*((float *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k), 3);
        if (k < tensor->shape[2]-1) {
          printf(" ");
        }
      }
      printf(" ]");
    }
    printf("]");
    if (i < tensor->shape[0]-1) {
      printf("\n");
    }
  }
  printf("]\n");
}

/**
 * Prints the content of a 4D floating-point data tensor.
 * 
 * @param tensor The 4D floating-point data tensor to print.
 */
void nn_print_tensor4d_f32(const Tensor4D_F32 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i == 0) {
      printf("[");
    }
    else {
      printf("\n [");
    }
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      if (j == 0) {
        printf("[");
      }
      else {
        printf("\n  [");
      }
      for (size_t k=0; k<tensor->shape[2]; k+=1) {
        if (k == 0) {
          printf("[");
        }
        else {
          printf("   [");
        }
        for (size_t l=0; l<tensor->shape[3]; l+=1) {
          nn_print_f32(*((float *)tensor->data + i*tensor->shape[1]*tensor->shape[2]*tensor->shape[3] + j*tensor->shape[2]*tensor->shape[3] + k*tensor->shape[3] + l), 3);
          if (l < tensor->shape[3]-1) {
            printf(" ");
          } 
        }
        printf(" ]");
        if (k < tensor->shape[2]-1) {
          printf("\n");
        }
      }
      printf("]");
      if (j < tensor->shape[1]-1) {
        printf("\n");
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
 * Checks if two 0D floating-point tensors are equal.
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
 * Checks if two 1D floating-point tensors are equal.
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
 * Checks if two 2D floating-point tensors are equal.
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

/**
 * Checks if two 3D floating-point tensors are equal.
 * 
 * @param a The first 3D tensor.
 * @param b The second 3D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals3d_f32(const Tensor3D_F32 *a, const Tensor3D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && a->shape[2] == b->shape[2], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1] * a->shape[2];
  for (size_t i = 0; i < n; i += 1) {
    if (!nn_equal_f32(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}

/**
 * Checks if two 4D floating-point tensors are equal.
 * 
 * @param a The first 4D tensor.
 * @param b The second 4D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals4d_f32(const Tensor4D_F32 *a, const Tensor4D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && a->shape[2] == b->shape[2] && a->shape[3] == b->shape[3], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1] * a->shape[2] * a->shape[3];
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
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
  #else  /* scalar implementation */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i += 1) {
      sum += x1->data[i] * x2->data[i];
    }
    y->data[0] = sum;
  #endif
}

/**
 * Performs a matrix multiplication of the matrices x1 and x2.
 * 
 * @param y The output tensor, shape (n, p)
 * @param x1 The first input tensor, shape (n, m)
 * @param x2 The second input tensor, shape (m, p)
 */
void nn_mm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) { 
  nn_assert(x1->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");

  const size_t n = x1->shape[0];
  const size_t m = x1->shape[1];
  const size_t p = x2->shape[1];
  
  for (size_t i = 0; i < n; i += 1) {
    #ifdef CONFIG_BACKEND_RISCV_V
      float *x1_row = x1->data + i * m;
      float *y_row = y->data + i * p;

      size_t vlmax = __riscv_vsetvlmax_e32m1();
      for (size_t j = 0; j < p; j += 1) {
        vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
        vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
        
        float *x1_ptr = x1_row;
        float *x2_ptr = x2->data + j;
        size_t k = m;
        
        while (k > 0) {
          size_t vl = __riscv_vsetvl_e32m1(k);
          vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_ptr, vl);
          vfloat32m1_t vec_x2 = __riscv_vlse32_v_f32m1(x2_ptr, p * sizeof(float), vl);
          vec_sum = __riscv_vfmacc_vv_f32m1(vec_sum, vec_x1, vec_x2, vl);
          
          x1_ptr += vl;
          x2_ptr += vl * p;
          k -= vl;
        }
        
        #ifdef CONFIG_DEBUG_RISCV_V_USE_REDOSUM
          vec_sum = __riscv_vfredosum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #else
          vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #endif
        y_row[j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
      }
    #else
      for (size_t j = 0; j < p; j += 1) {
        float sum = 0.f;
        for (size_t k = 0; k < m; k += 1) {
          sum += x1->data[i * m + k] * x2->data[k * p + j];
        }
        y->data[i * p + j] = sum;
      }
    #endif
  }
}


void nn_addmm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *c, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) { 
  nn_assert(x1->shape[1] == x2->shape[0], "Cannot perform Linear on tensors of different shapes");
  nn_assert(y->shape[0] == c->shape[0] && y->shape[1] == x2->shape[1], "Cannot perform Linear on tensors of different shapes");

  const size_t n = x1->shape[0];
  const size_t m = x1->shape[1];
  const size_t p = x2->shape[1];

  for (size_t i = 0; i < n; i += 1) {
    #ifdef CONFIG_BACKEND_RISCV_V
      float *x1_row = x1->data + i * m;
      float *y_row = y->data + i * p;

      size_t vlmax = __riscv_vsetvlmax_e32m1();
      for (size_t j = 0; j < p; j += 1) {
        vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
        vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
        
        float *x1_ptr = x1_row;
        float *x2_ptr = x2->data + j;
        size_t k = m;
        
        while (k > 0) {
          size_t vl = __riscv_vsetvl_e32m1(k);
          vfloat32m1_t vec_x1 = __riscv_vle32_v_f32m1(x1_ptr, vl);
          vfloat32m1_t vec_x2 = __riscv_vlse32_v_f32m1(x2_ptr, p * sizeof(float), vl);
          vec_sum = __riscv_vfmacc_vv_f32m1(vec_sum, vec_x1, vec_x2, vl);
          
          x1_ptr += vl;
          x2_ptr += vl * p;
          k -= vl;
        }
        
        #ifdef CONFIG_DEBUG_RISCV_V_USE_REDOSUM
          vec_sum = __riscv_vfredosum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #else
          vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #endif
        y_row[j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum) + c->data[i * p + j];
      }
      
      x1_row += m;
      y_row += p;
    #else
      for (size_t j = 0; j < p; j += 1) {
        float sum = 0.f;
        for (size_t k = 0; k < m; k += 1) {
          sum += x1->data[i * m + k] * x2->data[k * p + j];
        }
        y->data[i * p + j] = sum + c->data[i * p + j];
      }
    #endif
  }
}

void nn_linear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const Tensor2D_F32 *weight, const Tensor1D_F32 *bias) { 
  nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];
  
  float *x_batch_data = x->data;
  float *y_batch_data = y->data;

  for (size_t i = 0; i < batch_size; i += 1) {
    #ifdef CONFIG_BACKEND_RISCV_V
      float *x_data = x_batch_data;
      float *y_data = y_batch_data;

      size_t vlmax = __riscv_vsetvlmax_e32m1();

      for (size_t j = 0; j < out_features; j += 1) {
        vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
        vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);
        
        float *weight_row = weight->data + j * in_features;
        size_t n = in_features;
        
        while (n > 0) {
          size_t vl = __riscv_vsetvl_e32m1(n);
          vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x_data, vl);
          vfloat32m1_t vec_w = __riscv_vle32_v_f32m1(weight_row, vl);
          vec_sum = __riscv_vfmacc_vv_f32m1(vec_sum, vec_x, vec_w, vl);
          
          x_data += vl;
          weight_row += vl;
          n -= vl;
        }
        
        #ifdef CONFIG_DEBUG_RISCV_V_USE_REDOSUM
          vec_sum = __riscv_vfredosum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #else
          vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_zero, vlmax);
        #endif
        
        float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        if (bias) {
          sum += bias->data[j];
        }
        y_data[j] = sum;
        x_data = x_batch_data; // reset x_data pointer for next output feature
      }
      
      x_batch_data += in_features;
      y_batch_data += out_features;
    #else  /* scalar implementation */
      for (size_t j = 0; j < out_features; j += 1) {
        float sum = 0.f;
        for (size_t k = 0; k < in_features; k += 1) {
          sum += x->data[i * in_features + k] * weight->data[j * in_features + k];
        }
        if (bias) {
          sum += bias->data[j];
        }
        y->data[i * out_features + j] = sum;
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
  #else  /* scalar implementation */
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

  float *y_data = y->data;
  float *x_data = x->data;
  
  if (dim == 0) {
    for (size_t i = 0; i < y->shape[1]; i += 1) {
      size_t n = y->shape[0];
      size_t m = y->shape[1];
      float sum = 0.0f;
      for (size_t j = 0; j < n; j += 1) {
        sum += expf(x_data[j * m]);
      }

      for (size_t j = 0; j < n; j += 1) {
        y_data[j * m] = expf(x_data[j * m]) / sum;
      }

      x_data += 1;
      y_data += 1;
    }
  }
  else if (dim == 1) {
    // HACK: fix batch size
    for (size_t i = 0; i < y->shape[0]; i += 1) {
      size_t n = y->shape[1];
      float sum = 0.0f;
      for (size_t j = 0; j < n; j += 1) {
        sum += expf(x_data[j]);
      }

      for (size_t j = 0; j < n; j += 1) {
        y_data[j] = expf(x_data[j]) / sum;
      }

      x_data += n;
      y_data += n;
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

/**
 * Computes scaled dot product attention on query, key and value tensors.
 * 
 * Shape legend:
 * - N: batch size
 * - S: source sequence length
 * - L: target sequence length
 * - E: embedding dimension of the query and key
 * - Ev: embedding dimension of the value
 * - H: number of attention heads
 * 
 * @param y The output tensor, of shape (N, H, L, Ev).
 * @param query The query tensor, of shape (N, H, L, E).
 * @param key The key tensor, of shape (N, H, S, E).
 * @param value The value tensor, of shape (N, H, S, Ev).
 */
void nn_scaled_dot_product_attention_f32(Tensor4D_F32 *y, const Tensor4D_F32 *query, const Tensor4D_F32 *key, const Tensor4D_F32 *value) {
  nn_assert(query->shape[0] == key->shape[0] && query->shape[0] == value->shape[0], "Query, key, and value must have the same batch size");
  nn_assert(query->shape[1] == key->shape[1] && query->shape[1] == value->shape[1], "Query, key, and value must have the same sequence length");
  nn_assert(query->shape[2] == key->shape[2] && query->shape[2] == value->shape[2], "Query, key, and value must have the same head count");
  nn_assert(query->shape[3] == key->shape[3] && query->shape[3] == value->shape[3], "Query, key, and value must have the same embedding dimension");
  
  // L, S = query.size(-2), key.size(-2)
  size_t n = query->shape[0]; // batch size
  size_t h = query->shape[1]; // head count
  size_t s = key->shape[2]; // source sequence length
  size_t l = query->shape[2]; // target sequence length
  size_t e = query->shape[3]; // embedding dimension
  size_t ev = value->shape[3]; // value embedding dimension

  // scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
  float scale_factor = 1 / sqrt(query->shape[3]);
  // attn_bias = torch.zeros(L, S, dtype=query.dtype)

  // if is_causal:
  //     assert attn_mask is None
  //     temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
  //     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
  //     attn_bias.to(query.dtype)
  // if attn_mask is not None:
  //     if attn_mask.dtype == torch.bool:
  //         attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
  //     else:
  //         attn_bias += attn_mask

  // (n, hq, l, s) = (n, hq, l, e) @ (n, h, s, e).T
  size_t attn_weight_dims[4] = {n, h, l, s};
  Tensor4D_F32 *attn_weight = nn_tensor4d_f32(attn_weight_dims, NULL);
  
  size_t query_head_dims[2] = {l, e};
  size_t key_head_dims[2] = {l, e};
  size_t attn_weight_head_dims[2] = {l, s};
  size_t value_head_dims[2] = {s, ev};
  size_t y_head_dims[2] = {l, ev};

  for (size_t head = 0; head < h; head += 1) {
    Tensor2D_F32 *query_head = nn_as_tensor2d_f32(query_head_dims, (float *)query->data + head * l * e);
    Tensor2D_F32 *key_head = nn_as_tensor2d_f32(key_head_dims, (float *)key->data + head * s * e);
    Tensor2D_F32 *attn_weight_head = nn_as_tensor2d_f32(attn_weight_head_dims, (float *)attn_weight->data + head * l * s);
    Tensor2D_F32 *value_head = nn_as_tensor2d_f32(value_head_dims, (float *)value->data + head * s * ev);
    Tensor2D_F32 *y_head = nn_as_tensor2d_f32(y_head_dims, (float *)y->data + head * l * ev);

    // attn_weight = query @ key.transpose(-2, -1) * scale_factor
    nn_linear_f32(attn_weight_head, query_head, key_head, NULL);

    // attn_weight += attn_bias

    // attn_weight = torch.softmax(attn_weight, dim=-1)
    nn_softmax2d_f32(attn_weight_head, attn_weight_head, 1);

    // attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    // (n, hq, l, ev) = (n, hq, l, s) @ (n, h, s, ev)
    // attn_weight @ value
    nn_mm_f32(y_head, attn_weight_head, value_head);
  }
}


#endif // __NN_F32_H
