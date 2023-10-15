#ifndef NN_H
#define NN_H

#include <math.h>
#include <float.h>
#include <stddef.h>

typedef struct {
  size_t rows;
  size_t cols;
  float *data;
} Matrix;


void _assert(int condition, char *message) {
  if (!condition) {
    printf("Assertion failed: ");
    printf("%s\n", message);
    exit(1);
  }
}

void NN_initMatrix(Matrix *m, size_t rows, size_t cols) {
  m->rows = rows;
  m->cols = cols;
  m->data = malloc(rows * cols * sizeof(float));
}

void NN_matmul(Matrix *out, Matrix *a, Matrix *b) {
  _assert(a->cols == b->rows, "matmul: input dimension mismatch");
  _assert(out->rows == a->rows, "matmul: output dimension mismatch");
  _assert(out->cols == b->cols, "matmul: output dimension mismatch");
  for (size_t i = 0; i < a->rows; i++) {
    for (size_t j = 0; j < b->cols; j++) {
      float sum = 0;
      for (size_t k = 0; k < a->cols; k++) {
        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
      out->data[i * out->cols + j] = sum;
    }
  }
}

void NN_matadd(Matrix *out, Matrix *a, Matrix *b) {
  _assert(a->rows == b->rows, "matadd: dimension mismatch");
  _assert(a->cols == b->cols, "matadd: dimension mismatch");
  for (size_t i = 0; i < a->rows; i++) {
    for (size_t j = 0; j < a->cols; j++) {
      out->data[i * out->cols + j] = a->data[i * a->cols + j] + b->data[i * b->cols + j];
    }
  }
}

void NN_linear(Matrix *out, Matrix *weight_transposed, Matrix *bias, Matrix *input) {
  NN_matmul(out, input, weight_transposed);
  NN_matadd(out, out, bias);
}

void NN_transpose(Matrix *out, Matrix *a) {
  for (size_t i = 0; i < a->rows; i++) {
    for (size_t j = 0; j < a->cols; j++) {
      out->data[j * out->cols + i] = a->data[i * a->cols + j];
    }
  }
}

void NN_concatenate(Matrix *out, Matrix *a, Matrix *b) {
  for (size_t i = 0; i < a->cols; i++) {
    out->data[i] = a->data[i];
  }
  for (size_t i = 0; i < b->cols; i++) {
    out->data[a->cols + i] = b->data[i];
  }
}

void NN_logSoftmax(Matrix *out, Matrix *a) {
  float sum = 0;
  for (size_t i = 0; i < a->cols; i++) {
    sum += exp(a->data[i]);
  }
  for (size_t i = 0; i < a->cols; i++) {
    out->data[i] = log(exp(a->data[i]) / sum);
  }
}

void printDouble(double v, int decimalDigits) {
  int i = 1;
  int intPart, fractPart;
  for (;decimalDigits!=0; i*=10, decimalDigits--);
  intPart = (int)v;
  fractPart = (int)((v-(double)(int)v)*i);
  if(fractPart < 0) fractPart *= -1;
  printf("%i.%i", intPart, fractPart);
}

void printShape(Matrix *a) {
  printf("(%d, %d)\n", a->rows, a->cols);
}

void printMatrix(Matrix *a) {
  for (size_t i = 0; i < a->rows; i++) {
    for (size_t j = 0; j < a->cols; j++) {
      printDouble(a->data[i * a->cols + j], 2);
      printf(" ");
    }
    printf("\n");
  }
  printf("\n");
}

size_t argmax(Matrix *a) {
  int max_index = 0;
  float max_value = a->data[0];
  for (size_t i = 1; i < a->cols; i++) {
    if (a->data[i] > max_value) {
      max_index = i;
      max_value = a->data[i];
    }
  }
  return max_index;
}

#endif  // NN_H
