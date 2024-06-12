#ifndef __NN_TENSOR
#define __NN_TENSOR

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "nn_float16.h"

#define MAX_DIMS            4

// #define F32(ptr)              (*((float *)(ptr)))
// #define F64(ptr)              (*((double *)(ptr)))
// #define I8(ptr)               (*((int8_t *)(ptr)))
// #define I16(ptr)              (*((int16_t *)(ptr)))


typedef enum {
  DTYPE_I8,
  DTYPE_I16,
  DTYPE_I32,
  DTYPE_I64,
  DTYPE_I128,
  DTYPE_F8,
  DTYPE_F16,
  DTYPE_F32,
  DTYPE_F64,
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
    case DTYPE_I8:
      return sizeof(int8_t);
    case DTYPE_I16:
      return sizeof(int16_t);
    case DTYPE_I32:
      return sizeof(int32_t);
    case DTYPE_I64:
      return sizeof(int64_t);
    case DTYPE_F32:
      return sizeof(float);
    case DTYPE_F64:
      return sizeof(double);
    default:
      printf("[WARNING] Unsupported data type: %d\n", dtype);
      return 0;
  }
}

static inline const char *NN_getDataTypeName(DataType dtype) {
  switch (dtype) {
    case DTYPE_I8:
      return "INT8";
    case DTYPE_I16:
      return "INT16";
    case DTYPE_I32:
      return "INT32";
    case DTYPE_I64:
      return "INT64";
    case DTYPE_F32:
      return "FLOAT32";
    case DTYPE_F64:
      return "FLOAT64";
    default:
      return "UNKNOWN";
  }
}

/**
 * Returns if the tensor is a scalar
 * 
 * A scalar is a 1D tensor with a single element, i.e., shape = (1, )
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_isScalar(Tensor *tensor) {
  return tensor->ndim == 1 && tensor->shape[0] == 1;
}

/**
 * Returns if the tensor is a vector
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_isVector(Tensor *tensor) {
  return tensor->ndim == 1;
}

/**
 * Returns if the tensor is a matrix
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_isMatrix(Tensor *tensor) {
  return tensor->ndim == 2;
}

/**
 * Returns if the tensor is a 3D tensor
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_is3D(Tensor *tensor) {
  return tensor->ndim == 3;
}

/**
 * Returns if the tensor is a 4D tensor
 * 
 * @param tensor: the target tensor
 */
static inline uint8_t NN_is4D(Tensor *tensor) {
  return tensor->ndim == 4;
}

/**
 * Frees the memory allocated for the tensor data
 * 
 * @param tensor: the target tensor
 */
static inline void NN_freeTensorData(Tensor *tensor) {
  free(tensor->data);
}

/**
 * Frees the memory allocated for the tensor
 * 
 * @param tensor: the target tensor
 */
static inline void NN_deleteTensor(Tensor *tensor) {
  free(tensor);
}

/**
 * Fills the tensor with the specified value.
 * 
 * @param tensor: the input tensor
 * @param value: scalar value
 */
static inline void NN_fill_F32(Tensor *tensor, float value) {
  assert(tensor->dtype == DTYPE_F32);
  
  for (size_t i = 0; i < tensor->size; i += 1) {
    ((float *)tensor->data)[i] = value;
  }
}

static inline void NN_fill_I32(Tensor *tensor, int32_t value) {
  assert(tensor->dtype == DTYPE_I32);
  
  for (size_t i = 0; i < tensor->size; i += 1) {
    ((int32_t *)tensor->data)[i] = value;
  }
}

static inline void NN_fill_I8(Tensor *tensor, int8_t value) {
  assert(tensor->dtype == DTYPE_I8);
  
  for (size_t i = 0; i < tensor->size; i += 1) {
    ((int8_t *)tensor->data)[i] = value;
  }
}


/**
 * Initialize a given tensor
 * 
 * The memory is initialized in C order, i.e., the last dimension is contiguous.
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @param data: pointer to data, if NULL, the data will be allocated
 */
void NN_initTensor(Tensor *tensor, size_t ndim, const size_t *shape, DataType dtype, void *data);

/**
 * Create a new tensor
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @param data: pointer to data, if NULL, the data will be allocated
 * @return Tensor
*/
Tensor *NN_tensor(size_t ndim, const size_t *shape, DataType dtype, void *data);

/**
 * Returns a tensor filled with the scalar value 0.
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @return Tensor
 */
Tensor *NN_zeros(size_t ndim, const size_t *shape, DataType dtype);

/**
 * Returns a tensor filled with the scalar value 1.
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @return Tensor
 */
Tensor *NN_ones(size_t ndim, const size_t *shape, DataType dtype);

/**
 * Returns a tensor filled with random numbers from a uniform distribution.
 * 
 * The range of the random number is dependent on the data type:
 * - For Float32, the range is [0, 1]
 * - For Int8, the range is [0, 255]
 * - For Int32, the range is [0, RAND_MAX]
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @return Tensor
 */
Tensor *NN_rand(size_t ndim, const size_t *shape, DataType dtype);

/**
 * Returns this tensor cast to the type of the given tensor.
 * 
 * This is equivalent to NN_copy() if the data types are the same.
 * 
 * @param out: the output tensor
 * @param in: the input tensor
 */
void NN_asType(Tensor *out, Tensor *in);


#endif // __NN_TENSOR