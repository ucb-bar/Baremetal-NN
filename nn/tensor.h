#ifndef __NN_TENSOR
#define __NN_TENSOR

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "float16.h"

#define MAX_DIMS            4

// #define F32(ptr)              (*((float *)(ptr)))
// #define F64(ptr)              (*((double *)(ptr)))
// #define I8(ptr)               (*((int8_t *)(ptr)))
// #define I16(ptr)              (*((int16_t *)(ptr)))


typedef enum {
  DTYPE_U8,
  DTYPE_I8,
  DTYPE_U16,
  DTYPE_I16,
  DTYPE_U32,
  DTYPE_I32,
  // DTYPE_I64,
  // DTYPE_I128,
  // DTYPE_F8,
  DTYPE_F16,
  DTYPE_F32,
  // DTYPE_F64,
} DataType;

typedef struct {
  DataType dtype;             /** datatype */
  size_t ndim;                /** number of dimensions */
  size_t size;                /** number of elements */
  size_t shape[MAX_DIMS];     /** shape of tensor */
  void *data;                 /** data */
} Tensor;


static inline size_t NN_sizeof(DataType dtype) {
  switch (dtype) {
    case DTYPE_U8:
      return sizeof(uint8_t);
    case DTYPE_I8:
      return sizeof(int8_t);
    case DTYPE_U16:
      return sizeof(uint16_t);
    case DTYPE_I16:
      return sizeof(int16_t);
    case DTYPE_U32:
      return sizeof(uint32_t);
    case DTYPE_I32:
      return sizeof(int32_t);
    // case DTYPE_I64:
    //   return sizeof(int64_t);
    case DTYPE_F16:
      return sizeof(float16_t);
    case DTYPE_F32:
      return sizeof(float);
    // case DTYPE_F64:
    //   return sizeof(double);
    default:
      printf("[WARNING] Unsupported data type: %d\n", dtype);
      return 0;
  }
}

static inline const char *NN_get_datatype_name(DataType dtype) {
  switch (dtype) {
    case DTYPE_U8:
      return "UINT8";
    case DTYPE_I8:
      return "INT8";
    case DTYPE_U16:
      return "UINT16";
    case DTYPE_I16:
      return "INT16";
    case DTYPE_U32:
      return "UINT32";
    case DTYPE_I32:
      return "INT32";
    // case DTYPE_I64:
    //   return "INT64";
    case DTYPE_F16:
      return "FLOAT16";
    case DTYPE_F32:
      return "FLOAT32";
    // case DTYPE_F64:
    //   return "FLOAT64";
    default:
      return "UNKNOWN";
  }
}

/**
 * Returns if the tensor is a scalar
 * 
 * A scalar is a 0D tensor with a single element
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_is_scalar(Tensor *tensor) {
  return tensor->ndim == 0;
}

/**
 * Returns if the tensor is a vector
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_is_vector(Tensor *tensor) {
  return tensor->ndim == 1;
}

/**
 * Returns if the tensor is a matrix
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_is_matrix(Tensor *tensor) {
  return tensor->ndim == 2;
}

/**
 * Returns if the tensor is a 3D tensor
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_is_3d(Tensor *tensor) {
  return tensor->ndim == 3;
}

/**
 * Returns if the tensor is a 4D tensor
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_is_4d(Tensor *tensor) {
  return tensor->ndim == 4;
}

/**
 * Frees the memory allocated for the tensor data
 * 
 * @param tensor: the target tensor
 */
static inline void NN_free_tensor_data(Tensor *tensor) {
  free(tensor->data);
}

/**
 * Frees the memory allocated for the tensor
 * 
 * @param tensor: the target tensor
 */
static inline void NN_delete_tensor(Tensor *tensor) {
  free(tensor);
}



#endif // __NN_TENSOR