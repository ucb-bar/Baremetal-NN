#ifndef __NN_H
#define __NN_H

#include <math.h>
#include <float.h>
#include <stddef.h>

typedef struct {
  size_t rows;
  size_t cols;
  float *data;
} Matrix;


/*
 * ====== Utility Functions ======
 */

void NN_assert(int condition, char *message) {
  if (!condition) {
    printf("Assertion failed: ");
    printf("%s\n", message);
    exit(1);
  }
}

/*
 * ====== Print Functions ======
 *
 * These functions assumes that printf is available.
 */

void NN_printFloat(float v, int16_t num_digits) {
  int32_t scale = 1;
  int32_t integer_part, fractional_part;
  while (num_digits != 0) {
    scale *= 10;
    num_digits -= 1;
  }
  integer_part = (int32_t)v;
  fractional_part = (int32_t)((v-(float)(int32_t)v)*scale);
  if (fractional_part < 0) {
    fractional_part *= -1;
  }
  printf("%i.%i", integer_part, fractional_part);
}

void NN_printShape(Matrix *a) {
  printf("(%d, %d)\n", a->rows, a->cols);
}

void NN_printMatrix(Matrix *a) {
  for (size_t i = 0; i < a->rows; i++) {
    for (size_t j = 0; j < a->cols; j++) {
      NN_printFloat(a->data[i * a->cols + j], 2);
      printf(" ");
    }
    printf("\n");
  }
  printf("\n");
}

/*
 * ====== Math Functions ======
 */
void NN_initMatrix(Matrix *m, size_t rows, size_t cols) {
  m->rows = rows;
  m->cols = cols;
  m->data = malloc(rows * cols * sizeof(float));
}

void NN_matmul(Matrix *out, Matrix *a, Matrix *b) {
  NN_assert(a->cols == b->rows, "matmul: dimension mismatch");
  NN_assert(out->rows == a->rows, "matmul: dimension mismatch");
  NN_assert(out->cols == b->cols, "matmul: dimension mismatch");
  for (size_t i = 0; i < a->rows; i += 1) {
    for (size_t j = 0; j < b->cols; j += 1) {
      float sum = 0;
      for (size_t k = 0; k < a->cols; k += 1) {
        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
      out->data[i * out->cols + j] = sum;
    }
  }
}

void NN_matadd(Matrix *out, Matrix *a, Matrix *b) {
  NN_assert(a->rows == b->rows, "matadd: dimension mismatch");
  NN_assert(a->cols == b->cols, "matadd: dimension mismatch");
  for (size_t i = 0; i < a->rows; i += 1) {
    for (size_t j = 0; j < a->cols; j += 1) {
      out->data[i * out->cols + j] = a->data[i * a->cols + j] + b->data[i * b->cols + j];
    }
  }
}

void NN_transpose(Matrix *out, Matrix *a) {
  for (size_t i = 0; i < a->rows; i += 1) {
    for (size_t j = 0; j < a->cols; j += 1) {
      out->data[j * out->cols + i] = a->data[i * a->cols + j];
    }
  }
}

void NN_concatenate(Matrix *out, Matrix *a, Matrix *b) {
  for (size_t i = 0; i < a->cols; i += 1) {
    out->data[i] = a->data[i];
  }
  for (size_t i = 0; i < b->cols; i += 1) {
    out->data[a->cols + i] = b->data[i];
  }
}

size_t NN_argmax(Matrix *a) {
  int max_index = 0;
  float max_value = a->data[0];
  for (size_t i = 1; i < a->cols; i += 1) {
    if (a->data[i] > max_value) {
      max_index = i;
      max_value = a->data[i];
    }
  }
  return max_index;
}

/*
 * ====== Operators ======
 */

void NN_linear(Matrix *out, Matrix *weight, Matrix *bias, Matrix *input) {
  NN_matmul(out, input, weight);
  NN_matadd(out, out, bias);
}

void NN_logSoftmax(Matrix *out, Matrix *a) {
  float sum = 0;
  for (size_t i = 0; i < a->cols; i += 1) {
    sum += exp(a->data[i]);
  }
  for (size_t i = 0; i < a->cols; i += 1) {
    out->data[i] = log(exp(a->data[i]) / sum);
  }
}


#endif  // __NN_H
