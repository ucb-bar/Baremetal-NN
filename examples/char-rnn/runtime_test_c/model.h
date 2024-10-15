#ifndef __MODEL_H
#define __MODEL_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>

#include "nn.h"
#include "weights.h"

static Matrix i2h_weight_transposed = {
  .rows = I2H_WEIGHT_TRANSPOSED_ROWS,
  .cols = I2H_WEIGHT_TRANSPOSED_COLS,
  .data = I2H_WEIGHT_TRANSPOSED_DATA
};
static Matrix i2h_bias = {
  .rows = I2H_BIAS_ROWS,
  .cols = I2H_BIAS_COLS,
  .data = I2H_BIAS_DATA
};
static Matrix h2o_weight_transposed = {
  .rows = H2O_WEIGHT_TRANSPOSED_ROWS,
  .cols = H2O_WEIGHT_TRANSPOSED_COLS,
  .data = H2O_WEIGHT_TRANSPOSED_DATA
};
static Matrix h2o_bias = {
  .rows = H2O_BIAS_ROWS,
  .cols = H2O_BIAS_COLS,
  .data = H2O_BIAS_DATA
};


static void forward(Matrix *output, Matrix *hidden, Matrix *input) {
  // Input
  Matrix *input_out = input;
  // Linear
  nn_linear(hidden, &i2h_weight_transposed, &i2h_bias, input_out);
  // Linear
  nn_linear(output, &h2o_weight_transposed, &h2o_bias, hidden);
  // Log Softmax
  nn_logSoftmax(output, output);
}

#endif  // __MODEL_H
