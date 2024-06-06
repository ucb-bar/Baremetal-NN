#ifndef __NN_TENSOR
#define __NN_TENSOR

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_DIMS            4


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

static inline void NN_freeTensorData(Tensor *t) {
  free(t->data);
}

static inline void NN_deleteTensor(Tensor *t) {
  free(t);
}


/**
 * Initialize a tensor
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: DataType
 * @param data: pointer to data, if NULL, the data will be allocated
 */
void NN_initTensor(Tensor *t, size_t ndim, size_t *shape, DataType dtype, void *data);

Tensor *NN_tensor(size_t ndim, size_t *shape, DataType dtype, void *data);

Tensor *NN_zeros(size_t ndim, size_t *shape, DataType dtype);

Tensor *NN_ones(size_t ndim, size_t *shape, DataType dtype);

Tensor *NN_rand(size_t ndim, size_t *shape, DataType dtype);

/**
 * Convert tensor data type
 * 
 * @param t: input tensor
 * @param dtype: target data type
 */
void NN_asType(Tensor *t, DataType dtype);


#endif // __NN_TENSOR