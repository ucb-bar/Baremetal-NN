#ifndef __NN_H
#define __NN_H

#include <assert.h>

#include "nn_types.h"
#include "nn_add.h"
#include "nn_clip.h"
#include "nn_elu.h"
#include "nn_linear.h"
#include "nn_matmul.h"
#include "nn_relu.h"
#include "nn_transpose.h"


// http://elm-chan.org/junk/32bit/binclude.html
#define INCLUDE_FILE(section, filename, symbol) asm (\
    ".section "#section"\n"                   /* Change section */\
    ".balign 4\n"                             /* Word alignment */\
    ".global "#symbol"_start\n"               /* Export the object start address */\
    ".global "#symbol"_data\n"                /* Export the object address */\
    #symbol"_start:\n"                        /* Define the object start address label */\
    #symbol"_data:\n"                         /* Define the object label */\
    ".incbin \""filename"\"\n"                /* Import the file */\
    ".global "#symbol"_end\n"                 /* Export the object end address */\
    #symbol"_end:\n"                          /* Define the object end address label */\
    ".balign 4\n"                             /* Word alignment */\
    ".section \".text\"\n")                   /* Restore section */



void NN_assert(int condition, char *message) {
  if (!condition) {
    printf("Assertion failed: ");
    printf("%s\n", message);
    exit(1);
  }
}


/**
 * Create a tensor
 * 
 * @param ndim: number of dimensions
 * @param shape: shape of tensor
 * @param dtype: DataType
 */
void NN_initTensor(Tensor *t, size_t ndim, size_t *shape, DataType dtype, void *data) {
  t->ndim = ndim;
  t->dtype = dtype;
  t->data = data;

  // set shape
  for (size_t i = 0; i < ndim; i += 1) {
    t->shape[i] = shape[i];
  }
  for (size_t i = ndim; i < MAX_DIMS; i += 1) {
    t->shape[i] = 0;
  }

  // set stride
  t->stride[0] = NN_sizeof(dtype);
  for (size_t i = 1; i <= ndim; i += 1) {
    t->stride[i] = t->stride[i-1] * t->shape[ndim-i];
  }
  
  // calculate size (number of elements)
  t->size = 1;
  for (size_t i = 0; i < ndim; i += 1) {
    t->size *= t->shape[i];
  }
}

void NN_freeTensor(Tensor *t) {
  free(t->data);
  free(t);
}


/*
 * ====== Print Functions ======
 *
 * These functions assumes that printf is available.
 */
void NN_printFloat(float v, int16_t num_digits) {
  if (v < 0) {
    printf("-");  // Print the minus sign for negative numbers
    v = -v;        // Make the number positive for processing
  }

  // Calculate the integer part of the number
  long int_part = (long)v;
  float fractional_part = v - int_part;

  // Count the number of digits in the integer part
  long temp = int_part;
  int int_digits = (int_part == 0) ? 1 : 0; // Handle zero as a special case
  while (temp > 0) {
    int_digits++;
    temp /= 10;
  }

  // Print the integer part
  printf("%ld", int_part);

  // Calculate the number of fractional digits we can print
  int fractional_digits = num_digits - int_digits;
  if (fractional_digits > 0) {
    printf("."); // Print the decimal point

    // Handle the fractional part
    while (fractional_digits-- > 0) {
      fractional_part *= 10;
      int digit = (int)(fractional_part);
      printf("%d", digit);
      fractional_part -= digit;
    }
  }
}

void NN_printShape(Tensor *t) {
  printf("(");
  for (size_t i = 0; i < t->ndim; i += 1) {
    printf("%d", (int)t->shape[i]);
    if (i < t->ndim-1) {
      printf(", ");
    }
  }
  printf(")\n");
}

void NN_printf(Tensor *t) {
  // print data with torch.Tensor style
  if (t->ndim == 1) {
    printf("[");
    for (size_t i=0; i<t->shape[0]; i+=1) {
      switch (t->dtype) {
        case DTYPE_I8:
          printf("%d", ((int8_t *)t->data)[i]);
          break;
        case DTYPE_I32:
          printf("%ld", (size_t)((int32_t *)t->data)[i]);
          break;
        case DTYPE_F32:
          NN_printFloat(((float *)t->data)[i], 4);
          break;
      }
      if (i < t->shape[0]-1) {
        printf(" ");
      }
    }
    printf("]");
    printf("\n");
    return;
  }

  printf("[");
  for (size_t i=0; i<t->shape[0]; i+=1) {
    if (i != 0) {
      printf(" ");
    }
    printf("[");
    for (size_t j=0; j<t->shape[1]; j+=1) {
      switch (t->dtype) {
        case DTYPE_I8:
          printf("%d", ((int8_t *)t->data)[i*t->shape[1]+j]);
          break;
        case DTYPE_I32:
          printf("%ld", (size_t)((int32_t *)t->data)[i*t->shape[1]+j]);
          break;
        case DTYPE_F32:
          NN_printFloat(((float *)t->data)[i*t->shape[1]+j], 4);
          break;
      }
      if (j < t->shape[1]-1) {
        printf(" ");
      }
    }
    printf("]");
    if (i < t->shape[0]-1) {
      printf("\n");
    }
  }
  printf("]");
  printf("\n");
}

/**
 * Convert tensor data type
 * 
 * @param t: input tensor
 * @param dtype: target data type
 */
void NN_asType(Tensor *t, DataType dtype) {
  if (t->dtype == dtype) {
    return;
  }
  if (t->dtype == DTYPE_I32 && dtype == DTYPE_F32) {
    for (size_t i = 0; i<t->size; i+=1) {
      ((float *)t->data)[i] = (float)((int32_t *)t->data)[i];
    }
    t->dtype = DTYPE_F32;
    return;
  }
  if (t->dtype == DTYPE_I32 && dtype == DTYPE_I8) {
    for (size_t i = 0; i<t->size; i+=1) {
      ((int8_t *)t->data)[i] = (int8_t)((int32_t *)t->data)[i];
    }
    t->dtype = DTYPE_I8;
    return;
  }

  if (t->dtype == DTYPE_F32 && dtype == DTYPE_I32) {
    for (size_t i = 0; i<t->size; i+=1) {
      ((int32_t *)t->data)[i] = (int32_t)((float *)t->data)[i];
    }
    t->dtype = DTYPE_I32;
    return;
  }

  printf("Cannot convert data type from %s to %s\n", NN_getDataTypeName(t->dtype), NN_getDataTypeName(dtype));
}



/**
 * Copies values from one tensor to another
 * 
 * @param dst: destination tensor
 * @param src: source tensor
 */
void NN_copyTo(Tensor *dst, Tensor *src) {
  assert(dst->shape[0] == src->shape[0]);
  assert(dst->shape[1] == src->shape[1]);
  assert(dst->dtype == src->dtype);

  switch (dst->dtype) {
    case DTYPE_I8:
      for (size_t i = 0; i<dst->size; i+=1) {
        ((int8_t *)dst->data)[i] = ((int8_t *)src->data)[i];
      }
      break;
    case DTYPE_I32:
      for (size_t i = 0; i<dst->size; i+=1) {
        ((int32_t *)dst->data)[i] = ((int32_t *)src->data)[i];
      }
      break;
    case DTYPE_F32:
      for (size_t i = 0; i<dst->size; i+=1) {
        ((float *)dst->data)[i] = ((float *)src->data)[i];
      }
      break;
    default:
      printf("Unsupported data type\n");
  }
}





#endif // __NN_H