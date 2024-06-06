
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

void NN_printShape(Tensor *tensor) {
  printf("(");
  for (size_t i = 0; i < tensor->ndim; i += 1) {
    printf("%d", (int)tensor->shape[i]);
    if (i < tensor->ndim-1) {
      printf(", ");
    }
  }
  printf(")");
}

/**
 * Print the tensor with torch.Tensor style
 */
void NN_printf(Tensor *tensor) {
  printf("[");

  switch (tensor->ndim) {
    case 1:
      for (size_t i=0; i<tensor->shape[0]; i+=1) {
        switch (tensor->dtype) {
          case DTYPE_I8:
            printf("%d", *((int8_t *)tensor->data + i));
            break;
          case DTYPE_I32:
            printf("%ld", NN_get_I32_1D(tensor, i));
            break;
          case DTYPE_F32:
            NN_printFloat(NN_get_F32_1D(tensor, i), 4);
            break;
        }
        if (i < tensor->shape[0]-1) {
          printf(" ");
        }
      }
      break;
    
    case 2:
      for (size_t i=0; i<tensor->shape[0]; i+=1) {
        if (i != 0) {
          printf(" ");
        }
        printf("[");
        for (size_t j=0; j<tensor->shape[1]; j+=1) {
          switch (tensor->dtype) {
            case DTYPE_I8:
              printf("%d", *((int8_t *)tensor->data + i*tensor->shape[1] + j));
              break;
            case DTYPE_I32:
              printf("%ld", NN_get_I32_2D(tensor, i, j));
              break;
            case DTYPE_F32:
              NN_printFloat(NN_get_F32_2D(tensor, i, j), 4);
              break;
          }
          if (j < tensor->shape[1]-1) {
            printf(" ");
          }
        }
        printf("]");
        if (i < tensor->shape[0]-1) {
          printf("\n");
        }
      }
      break;

    case 3:
      for (size_t i=0; i<tensor->shape[0]; i+=1) {
        if (i != 0) {
          printf("\n");
        }
        printf("[");
        for (size_t j=0; j<tensor->shape[1]; j+=1) {
          if (j != 0) {
            printf(" ");
          }
          printf("[");
          for (size_t k=0; k<tensor->shape[2]; k+=1) {
            switch (tensor->dtype) {
              case DTYPE_I8:
                printf("%d", *((int8_t *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k));
                break;
              case DTYPE_I32:
                printf("%ld", NN_get_I32_3D(tensor, i, j, k));
                break;
              case DTYPE_F32:
                NN_printFloat(NN_get_F32_3D(tensor, i, j, k), 4);
                break;
            }
            if (k < tensor->shape[2]-1) {
              printf(" ");
            }
          }
          printf("]");
        }
        printf("]");
      }
  }
      
  printf("]");
  printf("\n");
}
