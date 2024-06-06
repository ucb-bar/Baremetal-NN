#ifndef __NN_TENSOR
#define __NN_TENSOR

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

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
  
  // strides[0] = strides[1] * shape[1]
  // strides[1] = strides[2] * shape[2]
  // ...
  // strides[ndim-1] = sizeof(dtype)
  size_t strides[MAX_DIMS];   /** strides, in bytes */

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

static inline uint8_t NN_isScalar(Tensor *tensor) {
  return tensor->ndim == 1 && tensor->shape[0] == 1;
}

static inline uint8_t NN_isVector(Tensor *tensor) {
  return tensor->ndim == 1;
}

static inline uint8_t NN_isMatrix(Tensor *tensor) {
  return tensor->ndim == 2;
}

static inline uint8_t NN_is3D(Tensor *tensor) {
  return tensor->ndim == 3;
}

static inline int32_t NN_get_I32_1D(Tensor *tensor, size_t i) {
  return *((int32_t *)((uint8_t *)tensor->data + i * tensor->strides[0]));
}

static inline void NN_set_I32_1D(Tensor *tensor, size_t i, int32_t value) {
  *((int32_t *)((uint8_t *)tensor->data + i * tensor->strides[0])) = value;
}

static inline int32_t NN_get_I32_2D(Tensor *tensor, size_t i0, size_t i1) {
  return *((int32_t *)((uint8_t *)tensor->data + i0 * tensor->strides[0] + i1 * tensor->strides[1]));
}

static inline void NN_set_I32_2D(Tensor *tensor, size_t i0, size_t i1, int32_t value) {
  *((int32_t *)((uint8_t *)tensor->data + i0 * tensor->strides[0] + i1 * tensor->strides[1])) = value;
}

static inline int32_t NN_get_I32_3D(Tensor *tensor, size_t i0, size_t i1, size_t i2) {
  return *((int32_t *)((uint8_t *)tensor->data + i0 * tensor->strides[0] + i1 * tensor->strides[1] + i2 * tensor->strides[2]));
}

static inline void NN_set_I32_3D(Tensor *tensor, size_t i0, size_t i1, size_t i2, int32_t value) {
  *((int32_t *)((uint8_t *)tensor->data + i0 * tensor->strides[0] + i1 * tensor->strides[1] + i2 * tensor->strides[2])) = value;
}

static inline float NN_get_F32_1D(Tensor *tensor, size_t i) {
  return *((float *)((uint8_t *)tensor->data + i * tensor->strides[0]));
}

static inline void NN_set_F32_1D(Tensor *tensor, size_t i, float value) {
  *((float *)((uint8_t *)tensor->data + i * tensor->strides[0])) = value;
}

static inline float NN_get_F32_2D(Tensor *tensor, size_t i0, size_t i1) {
  return *((float *)((uint8_t *)tensor->data + i0 * tensor->strides[0] + i1 * tensor->strides[1]));
}

static inline void NN_set_F32_2D(Tensor *tensor, size_t i0, size_t i1, float value) {
  *((float *)((uint8_t *)tensor->data + i0 * tensor->strides[0] + i1 * tensor->strides[1])) = value;
}

static inline float NN_get_F32_3D(Tensor *tensor, size_t i0, size_t i1, size_t i2) {
  return *((float *)((uint8_t *)tensor->data + i0 * tensor->strides[0] + i1 * tensor->strides[1] + i2 * tensor->strides[2]));
}

static inline void NN_set_F32_3D(Tensor *tensor, size_t i0, size_t i1, size_t i2, float value) {
  *((float *)((uint8_t *)tensor->data + i0 * tensor->strides[0] + i1 * tensor->strides[1] + i2 * tensor->strides[2])) = value;
}


/**
 * Frees the memory allocated for the tensor data
 */
static inline void NN_freeTensorData(Tensor *tensor) {
  free(tensor->data);
}

/**
 * Frees the memory allocated for the tensor
 */
static inline void NN_deleteTensor(Tensor *tensor) {
  free(tensor);
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
void NN_initTensor(Tensor *tensor, size_t ndim, size_t *shape, DataType dtype, void *data);

/**
 * Create a new tensor
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @param data: pointer to data, if NULL, the data will be allocated
 * @return Tensor
*/
Tensor *NN_tensor(size_t ndim, size_t *shape, DataType dtype, void *data);

/**
 * Returns a tensor filled with the scalar value 0.
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @return Tensor
 */
Tensor *NN_zeros(size_t ndim, size_t *shape, DataType dtype);

/**
 * Returns a tensor filled with the scalar value 1.
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: data type
 * @return Tensor
 */
Tensor *NN_ones(size_t ndim, size_t *shape, DataType dtype);

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
Tensor *NN_rand(size_t ndim, size_t *shape, DataType dtype);

/**
 * Returns this tensor cast to the type of the given tensor.
 * 
 * This is a no-op if the tensor is already of the correct type. 
 * 
 * @param t: input tensor
 * @param dtype: target data type
 */
void NN_asType(Tensor *t, DataType dtype);


#endif // __NN_TENSOR