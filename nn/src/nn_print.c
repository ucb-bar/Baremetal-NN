
#include "nn_print.h"


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
  printf(")");
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
