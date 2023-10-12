#include <float.h>
#include <stddef.h>

Matrix i2h_weight_t = {.rows = 32, .cols = 57, .data = I2H_WEIGHT_T_DATA};
Matrix i2h_bias = {.rows = 1, .cols = 32, .data = I2H_BIAS_DATA};
Matrix h2o_weight_t = {.rows = 32, .cols = 32, .data = H2O_WEIGHT_T_DATA};
Matrix h2o_bias = {.rows = 1, .cols = 32, .data = H2O_BIAS_DATA};


void forward(Matrix *output, Matrix* input) {
  // Linear
  NN_linear(output, i2h_weight_t, i2h_bias, input);
  // Linear
  NN_linear(output, h2o_weight_t, h2o_bias, input);
  // Log Softmax
  NN_logsoftmax(output, input);
}



