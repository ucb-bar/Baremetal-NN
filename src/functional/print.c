
#include "print.h"


void NN_print_f16(float16_t v, int16_t num_digits) {
  NN_print_f32(NN_half_to_float(v), num_digits);
}

void NN_print_f32(float v, int16_t num_digits) {
  if (isinf(v)) {
    if (signbit(v)) {
      printf("-inf");
    } else {
      printf("inf");
    }
    return;
  }
  
  if (v < 0) {
    printf("-");  // Print the minus sign for negative numbers
    v = -v;        // Make the number positive for processing
  }

  // Calculate the integer part of the number
  long int_part = (long)v;
  float fractional_part = v - int_part;

  // Print the integer part
  printf("%ld", int_part);

  if (num_digits > 0) {
    printf("."); // Print the decimal point
  }

  // Handle the fractional part
  while (num_digits > 0) {
    num_digits -= 1;
    fractional_part *= 10;
    int digit = (int)(fractional_part);
    printf("%d", digit);
    fractional_part -= digit;
  }
}

void NN_print_shape(const Tensor *tensor) {
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
void NN_printf(const Tensor *tensor) {
  printf("[");

  switch (tensor->ndim) {
    case 1:
      for (size_t i=0; i<tensor->shape[0]; i+=1) {
        switch (tensor->dtype) {
          case DTYPE_U8:
            NN_print_u8(*((uint8_t *)tensor->data + i));
            break;
          case DTYPE_I8:
            NN_print_i8(*((int8_t *)tensor->data + i));
            break;
          case DTYPE_U16:
            NN_print_u16(*((uint16_t *)tensor->data + i));
            break;
          case DTYPE_I16:
            NN_print_i16(*((int16_t *)tensor->data + i));
            break;
          case DTYPE_U32:
            NN_print_u32(*((uint32_t *)tensor->data + i));
            break;
          case DTYPE_I32:
            NN_print_i32(*((int32_t *)tensor->data + i));
            break;
          case DTYPE_F16:
            NN_print_f16(*((float16_t *)tensor->data + i), 3);
            break;
          case DTYPE_F32:
            NN_print_f32(*((float *)tensor->data + i), 4);
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
            case DTYPE_U8:
              NN_print_u8(*((uint8_t *)tensor->data + i*tensor->shape[1] + j));
              break;
            case DTYPE_I8:
              NN_print_i8(*((int8_t *)tensor->data + i*tensor->shape[1] + j));
              break;
            case DTYPE_U16:
              NN_print_u16(*((uint16_t *)tensor->data + i*tensor->shape[1] + j));
              break;
            case DTYPE_I16:
              NN_print_i16(*((int16_t *)tensor->data + i*tensor->shape[1] + j));
              break;
            case DTYPE_U32:
              NN_print_u32(*((uint32_t *)tensor->data + i*tensor->shape[1] + j));
              break;
            case DTYPE_I32:
              NN_print_i32(*((int32_t *)tensor->data + i*tensor->shape[1] + j));
              break;
            case DTYPE_F16:
              NN_print_f16(*((float16_t *)tensor->data + i*tensor->shape[1] + j), 3);
              break;
            case DTYPE_F32:
              NN_print_f32(*((float *)tensor->data + i*tensor->shape[1] + j), 4);
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
              case DTYPE_U8:
                NN_print_u8(*((uint8_t *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k));
                break;
              case DTYPE_I8:
                NN_print_i8(*((int8_t *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k));
                break;
              case DTYPE_U16:
                NN_print_u16(*((uint16_t *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k));
                break;
              case DTYPE_I16:
                NN_print_i16(*((int16_t *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k));
                break;
              case DTYPE_U32:
                NN_print_u32(*((uint32_t *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k));
                break;
              case DTYPE_I32:
                NN_print_i32(*((int32_t *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k));
                break;
              case DTYPE_F16:
                NN_print_f16(*((float16_t *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k), 3);
                break;
              case DTYPE_F32:
                NN_print_f32(*((float *)tensor->data + i*tensor->shape[1]*tensor->shape[2] + j*tensor->shape[2] + k), 4);
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
      break;

    case 4:
      for (size_t n = 0; n < tensor->shape[0]; n += 1) {
        if (n != 0) {
          printf("\n");
        }
        printf("[");
        for (size_t c = 0; c < tensor->shape[1]; c += 1) {
          if (c != 0) {
            printf(" ");
          }
          printf("[");
          for (size_t h = 0; h < tensor->shape[2]; h += 1) {
            if (h != 0) {
              printf(" ");
            }
            printf("[");
            for (size_t w = 0; w < tensor->shape[3]; w += 1) {
              switch (tensor->dtype) {
                case DTYPE_U8:
                  NN_print_u8(*((uint8_t *)tensor->data + n*tensor->shape[1]*tensor->shape[2]*tensor->shape[3] + c*tensor->shape[2]*tensor->shape[3] + h*tensor->shape[3] + w));
                  break;
                case DTYPE_I8:
                  NN_print_i8(*((int8_t *)tensor->data + n*tensor->shape[1]*tensor->shape[2]*tensor->shape[3] + c*tensor->shape[2]*tensor->shape[3] + h*tensor->shape[3] + w));
                  break;
                case DTYPE_U16:
                  NN_print_u16(*((uint16_t *)tensor->data + n*tensor->shape[1]*tensor->shape[2]*tensor->shape[3] + c*tensor->shape[2]*tensor->shape[3] + h*tensor->shape[3] + w));
                  break;
                case DTYPE_I16:
                  NN_print_i16(*((int16_t *)tensor->data + n*tensor->shape[1]*tensor->shape[2]*tensor->shape[3] + c*tensor->shape[2]*tensor->shape[3] + h*tensor->shape[3] + w));
                  break;
                case DTYPE_U32:
                  NN_print_u32(*((uint32_t *)tensor->data + n*tensor->shape[1]*tensor->shape[2]*tensor->shape[3] + c*tensor->shape[2]*tensor->shape[3] + h*tensor->shape[3] + w));
                  break;
                case DTYPE_I32:
                  NN_print_i32(*((int32_t *)tensor->data + n*tensor->shape[1]*tensor->shape[2]*tensor->shape[3] + c*tensor->shape[2]*tensor->shape[3] + h*tensor->shape[3] + w));
                  break;
                case DTYPE_F16:
                  NN_print_f16(*((float16_t *)tensor->data + n*tensor->shape[1]*tensor->shape[2]*tensor->shape[3] + c*tensor->shape[2]*tensor->shape[3] + h*tensor->shape[3] + w), 3);
                  break;
                case DTYPE_F32:
                  NN_print_f32(*((float *)tensor->data + n*tensor->shape[1]*tensor->shape[2]*tensor->shape[3] + c*tensor->shape[2]*tensor->shape[3] + h*tensor->shape[3] + w), 4);
                  break;
              }
              if (w < tensor->shape[3]-1) {
                printf(" ");
              }
            }
            printf("]");
            if (h < tensor->shape[2]-1) {
              printf("\n");
            }
          }
          printf("]");
          if (c < tensor->shape[1]-1) {
            printf("\n\n");
          }
        }
        printf("]");
      }
      break;
  }
      
  printf("]\n");
}
