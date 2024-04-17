#ifndef __NN_TYPES
#define __NN_TYPES

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_DIMS            3


typedef enum {
  DTYPE_I8,
  DTYPE_I32,
  DTYPE_F32,
} DataType;


typedef struct {
  DataType dtype;             /** datatype */
  size_t ndim;                /** number of dimensions */
  size_t size;                /** number of elements */
  
  size_t shape[MAX_DIMS];     /** shape of tensor */
  
  // stride[0] = sizeof(dtype)
  // stride[1] = stride[0] * shape[ndim-1]
  // stride[2] = stride[1] * shape[ndim-2]
  // ...
  size_t stride[MAX_DIMS+1];  /** stride, in bytes */

  void *data;                 /** data */
} Tensor;


static inline size_t NN_sizeof(DataType dtype) {
  switch (dtype) {
    case DTYPE_I8:
      return sizeof(int8_t);
    case DTYPE_I32:
      return sizeof(int32_t);
    case DTYPE_F32:
      return sizeof(float);
    default:
      printf("Unsupported data type\n");
      return 0;
  }
}

static inline const char *NN_getDataTypeName(DataType dtype) {
  switch (dtype) {
    case DTYPE_I8:
      return "INT8";
    case DTYPE_I32:
      return "INT32";
    case DTYPE_F32:
      return "FLOAT32";
    default:
      return "UNKNOWN";
  }
}

#endif // __NN_TYPES