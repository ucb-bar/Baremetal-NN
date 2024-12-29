/**
 * @file nn.h
 * @brief The Baremetal-NN Library
 * 
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

#ifndef __NN_F32_H
#define __NN_F32_H

/**
 * Tensor0D_F32
 * 
 * A 0D tensor (scalar) with a float data type.
 */
typedef struct {
  float data;
} Tensor0D_F32;


/**
 * Tensor1D_F32
 * 
 * A 1D tensor with a float data type.
 */
typedef struct {
  size_t shape[1];
  float *data;
} Tensor1D_F32;


/**
 * Tensor2D_F32
 * 
 * A 2D tensor with a float data type.
 */
typedef struct {
  size_t shape[2];
  float *data;
} Tensor2D_F32;

/**
 * nn_equal_f32
 * 
 * Checks if two floating-point numbers are equal within a relative error.
 * 
 * @param golden The expected value.
 * @param actual The actual value.
 * @param rel_err The relative error tolerance.
 * @return 1 if the numbers are equal within the relative error, 0 otherwise.
 */
static inline uint8_t nn_equal_f32(float golden, float actual, float rel_err) {
  return (fabs(actual - golden) < rel_err) || (fabs((actual - golden) / actual) < rel_err);
}


/* ======================================================================================================== */
/*                                           Tensor Creation                                                */
/* ======================================================================================================== */
/**
 * nn_tensor0d_f32
 * 
 * Creates a 0D tensor with type F32.
 * 
 * @param data The data to store in the tensor.
 */
Tensor0D_F32 *nn_tensor0d_f32(float data) {
  Tensor0D_F32 *tensor = (Tensor0D_F32 *)malloc(sizeof(Tensor0D_F32));
  tensor->data = data;
}

/**
 * nn_tensor1d_f32
 * 
 * Creates a 1D tensor with type F32.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor1D_F32 *nn_tensor1d_f32(size_t shape[1], const float *data) {
  Tensor1D_F32 *tensor = (Tensor1D_F32 *)malloc(sizeof(Tensor1D_F32));
  tensor->shape[0] = shape[0];

  size_t n_bytes = shape[0] * sizeof(float);
  tensor->data = (float *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
}

/**
 * nn_tensor2d_f32
 * 
 * Creates a 2D tensor with type F32.
 * 
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor2D_F32 *nn_tensor2d_f32(size_t shape[2], const float *data) {
  Tensor2D_F32 *tensor = (Tensor2D_F32 *)malloc(sizeof(Tensor2D_F32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];

  size_t n_bytes = shape[0] * shape[1] * sizeof(float);
  tensor->data = (float *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
}

Tensor0D_F32 *nn_zeros0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(0);
  return tensor;
}

Tensor1D_F32 *nn_zeros1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

Tensor2D_F32 *nn_zeros2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

Tensor0D_F32 *nn_ones0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(1);
  return tensor;
}

Tensor1D_F32 *nn_ones1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

Tensor2D_F32 *nn_ones2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

Tensor0D_F32 *nn_full0d_f32(float data) {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(data);
  return tensor;
}

Tensor1D_F32 *nn_full1d_f32(size_t shape[1], float data) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

Tensor2D_F32 *nn_full2d_f32(size_t shape[2], float data) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

Tensor0D_F32 *nn_rand0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(rand());
  return tensor;
}

Tensor1D_F32 *nn_rand1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
  }
  return tensor;
}

Tensor2D_F32 *nn_rand2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
  }
  return tensor;
}


/* ======================================================================================================== */
/*                                           Tensor Prints                                                  */
/* ======================================================================================================== */

/**
 * nn_print_f32
 * 
 * Prints a float.
 * 
 * @param v The float to print.
 * @param num_digits The number of decimal digits to print.
 */
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

/**
 * nn_print_tensor1d_f32
 * 
 * Prints the content of a 1D tensor with type F32.
 * 
 * @param tensor The 1D tensor to print.
 */
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

/**
 * nn_print_tensor2d_f32
 * 
 * Prints the content of a 2D tensor with type F32.
 * 
 * @param tensor The 2D tensor to print.
 */
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


/* ======================================================================================================== */
/*                                           Comparision                                                    */
/* ======================================================================================================== */
/**
 * nn_equals0d_f32
 * 
 * Checks if two 0D tensors with type F32 are equal.
 * 
 * @param a The first 0D tensor.
 * @param b The second 0D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals0d_f32(const Tensor0D_F32 *a, const Tensor0D_F32 *b, float rel_err) {
  return nn_equal_f32(a->data, b->data, rel_err);
}

/**
 * nn_equals1d_f32
 * 
 * Checks if two 1D tensors with type F32 are equal.
 * 
 * @param a The first 1D tensor.
 * @param b The second 1D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals1d_f32(const Tensor1D_F32 *a, const Tensor1D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    if (!nn_equal_f32(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}

/**
 * nn_equals2d_f32
 * 
 * Checks if two 2D tensors with type F32 are equal.
 * 
 * @param a The first 2D tensor.
 * @param b The second 2D tensor.
 * @param rel_err The relative error tolerance.
 * @return 1 if the tensors are equal within the relative error, 0 otherwise.
 */
uint8_t nn_equals2d_f32(const Tensor2D_F32 *a, const Tensor2D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (!nn_equal_f32(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}



/* ======================================================================================================== */
/*                                           Unary                                                          */
/* ======================================================================================================== */
void nn_max1d_f32(Tensor0D_F32 *y, const Tensor1D_F32 *x) {
  y->data = -FLT_MAX;
  size_t n = x->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    float val = x->data[i];
    y->data = val > y->data ? val : y->data;
  }
}

void nn_max2d_f32(Tensor0D_F32 *y, const Tensor2D_F32 *x) {
  y->data = -FLT_MAX;
  size_t n = x->shape[0] * x->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float val = x->data[i];
    y->data = val > y->data ? val : y->data;
  }
}

void nn_min1d_f32(Tensor0D_F32 *y, const Tensor1D_F32 *x) {
  y->data = FLT_MAX;
  size_t n = x->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    float val = x->data[i];
    y->data = val < y->data ? val : y->data;
  }
}

void nn_min2d_f32(Tensor0D_F32 *y, const Tensor2D_F32 *x) {
  y->data = FLT_MAX;
  size_t n = x->shape[0] * x->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float val = x->data[i];
    y->data = val < y->data ? val : y->data;
  }
}

/* ======================================================================================================== */
/*                                           Addition                                                       */
/* ======================================================================================================== */
/**
 * nn_add1d_f32
 * 
 * Adds x1 and x2 element-wise and stores the result in y.
 * 
 * y[i] = x1[i] + x2[i]
 * 
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor. 
 */
void nn_add1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x1->data[i] + x2->data[i]; 
  }
}

/**
 * nn_add2d_f32
 * 
 * Adds x1 and x2 element-wise and stores the result in y.
 * 
 * y[i][j] = x1[i][j] + x2[i][j]
 * 
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor. 
 */
void nn_add2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x1->data[i] + x2->data[i]; 
  }
}

void nn_addscalar1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] + scalar; 
  }
}

void nn_addscalar2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] + scalar; 
  }
}

/* ======================================================================================================== */
/*                                           Multiplication                                                 */
/* ======================================================================================================== */


void nn_mul1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x1->data[i] * x2->data[i]; 
  }
}

void nn_mul2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x1->data[i] * x2->data[i]; 
  }
}

void nn_mulscalar1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] * scalar; 
  }
}

void nn_mulscalar2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = x->data[i] * scalar; 
  }
}


/* ======================================================================================================== */
/*                                           MatMul                                                         */
/* ======================================================================================================== */
void nn_dot_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x1, const Tensor1D_F32 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot dot tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot dot tensors of different shapes");

  size_t n = y->shape[0];
  float sum = 0.0;
  for (size_t i = 0; i < n; i += 1) {
    sum += x1->data[i] * x2->data[i];
  }
  y->data[0] = sum;
}

void nn_mm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2) { 
  nn_assert(x1->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");

  const size_t batch_size = x1->shape[0];
  const size_t in_features = x1->shape[1];
  const size_t out_features = x2->shape[0];

  for (size_t i = 0; i < batch_size; i += 1) {
    for (size_t j = 0; j < out_features; j += 1) {
      float sum = 0.f;
      for (size_t k = 0; k < in_features; k += 1) {
        sum += x1->data[i * in_features + k] * x2->data[j * in_features + k];
      }
      y->data[i * out_features + j] = sum;
    }
  }
}


void nn_addmm_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const Tensor2D_F32 *weight, const Tensor1D_F32 *bias) { 
  nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  nn_assert(bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < out_features; j++) {
      float sum = 0.f;
      for (size_t k = 0; k < in_features; k++) {
        sum += x->data[i * in_features + k] * weight->data[j * in_features + k];
      }
      y->data[i * out_features + j] = sum + bias->data[j];
    }
  }
}



/* ======================================================================================================== */
/*                                           Non-linear                                                     */
/* ======================================================================================================== */
void nn_elu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, float alpha) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ELU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (x->data[i] > 0) {
      y->data[i] = x->data[i];
    }
    else {
      y->data[i] = alpha * (expf(x->data[i]) - 1.f);
    }
  }
}

void nn_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x->data[i];
    y->data[i] = x_val > 0 ? x_val : 0;
  }
}


void nn_softmax1d_f32(Tensor1D_F32 *y, const Tensor1D_F32 *x) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  float sum = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    sum += expf(x->data[i]);
  }

  for (size_t i = 0; i < n; i += 1) {
    y->data[i] = expf(x->data[i]) / sum;
  }
}

void nn_softmax2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, size_t dim) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  if (dim == 0) {
    for (size_t i = 0; i < y->shape[0]; i += 1) {
      size_t n = y->shape[1];
      float sum = 0.0f;
      for (size_t j = 0; j < n; j += 1) {
        sum += expf(x->data[i * n + j]);
      }

      for (size_t j = 0; j < n; j += 1) {
        y->data[i * n + j] = expf(x->data[i * n + j]) / sum;
      }
    }
  }
  else if (dim == 1) {
    // HACK: fix batch size
    size_t n = y->shape[1];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i += 1) {
      sum += expf(x->data[i]);
    }

    for (size_t i = 0; i < n; i += 1) {
      y->data[i] = expf(x->data[i]) / sum;
    }
  }
  else {
    nn_assert(0, "Invalid dimension for softmax");
  }
}



void nn_tanh2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x->data[i];
    y->data[i] = tanh(x_val);
  }
}




#endif // __NN_F32_H