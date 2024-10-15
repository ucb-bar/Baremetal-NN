#include "nn.h"

void nn_print_u8(uint8_t v) {
  printf("%d", v);
}

void nn_print_i8(int8_t v) {
  printf("%d", v);
}

void nn_print_u16(uint16_t v) {
  printf("%d", v);
}

void nn_print_i16(int16_t v) {
  printf("%d", v);
}

void nn_print_u32(uint32_t v) {
  printf("%ld", (size_t)v);
}

void nn_print_i32(int32_t v) {
  printf("%ld", (size_t)v);
}

void nn_print_f16(float16_t v, int16_t num_digits) {
  nn_print_f32(as_f32(v), num_digits);
}

void nn_print_f32(float v, int16_t num_digits) {
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


void nn_print_shape(size_t ndim, const size_t *shape) {
  printf("(");
  for (size_t i = 0; i < ndim; i += 1) {
    printf("%d", (int)shape[i]);
    if (i < ndim-1) {
      printf(", ");
    }
  }
  printf(")");
}


void nn_print_tensor1d_f16(const Tensor1D_F16 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    nn_print_f16(*((float16_t *)tensor->data + i), 3);
    if (i < tensor->shape[0]-1) {
      printf(" ");
    }
  }
  printf("]\n");
}

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

void nn_print_tensor2d_f16(const Tensor2D_F16 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i != 0) {
      printf(" ");
    }
    printf("[");
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      nn_print_f16(*((float16_t *)tensor->data + i*tensor->shape[1] + j), 3);
      if (j < tensor->shape[1]-1) {
        printf(" ");
      }
    }
    printf("]");
    if (i < tensor->shape[0]-1) {
      printf("\n");
    }
  }
  printf("]\n");
}

void nn_print_tensor2d_f32(const Tensor2D_F32 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i != 0) {
      printf(" ");
    }
    printf("[");
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      nn_print_f32(*((float *)tensor->data + i*tensor->shape[1] + j), 3);
      if (j < tensor->shape[1]-1) {
        printf(" ");
      }
    }
    printf("]");
    if (i < tensor->shape[0]-1) {
      printf("\n");
    }
  }
  printf("]\n");
}

// void nn_print_tensor3d_f16(const Tensor3D_F16 *tensor);

// void nn_print_tensor3d_f32(const Tensor3D_F32 *tensor);

// void nn_print_tensor4d_f16(const Tensor4D_F16 *tensor);

// void nn_print_tensor4d_f32(const Tensor4D_F32 *tensor);


