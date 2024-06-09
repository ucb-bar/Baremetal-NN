/**
 * @file main.c
 * 
 * A simple example demonstrating C = A * B + D
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

#define DIM   16


// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "../model.bin", weights);
extern uint8_t weights_data[];
extern size_t weights_start[];
extern size_t weights_end[];

typedef struct {
  Tensor *input;
  Tensor *conv1_weight;
  Tensor *conv1_bias;
  Tensor *conv1_out;
  Tensor *batchnorm1_weight;
  Tensor *batchnorm1_bias;
  Tensor *batchnorm1_running_mean;
  Tensor *batchnorm1_running_var;
  Tensor *batchnorm1_out;
} Model;

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  uint8_t *array_pointer = weights_data;

  model->input = NN_ones(4, (size_t[]){1, 1, 4, 4}, DTYPE_F32);

  model->conv1_weight = NN_tensor(4, (size_t[]){2, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 2 * 1 * 3 * 3 * sizeof(float);
  // model->conv1_bias = NN_zeros(1, (size_t[]){2, }, DTYPE_F32);
  // array_pointer += 1 * sizeof(float);
  model->conv1_out = NN_tensor(4, (size_t[]){1, 2, 4, 4}, DTYPE_F32, NULL);

  // model->batchnorm1_weight = NN_tensor(1, (size_t[]){2, }, DTYPE_F32, array_pointer);
  model->batchnorm1_weight = NN_ones(1, (size_t[]){2, }, DTYPE_F32);
  array_pointer += 2 * sizeof(float);
  // model->batchnorm1_bias = NN_tensor(1, (size_t[]){2, }, DTYPE_F32, array_pointer);
  model->batchnorm1_bias = NN_zeros(1, (size_t[]){2, }, DTYPE_F32);
  array_pointer += 2 * sizeof(float);
  // model->batchnorm1_running_mean = NN_tensor(1, (size_t[]){2, }, DTYPE_F32, array_pointer);
  model->batchnorm1_running_mean = NN_zeros(1, (size_t[]){2, }, DTYPE_F32);
  array_pointer += 2 * sizeof(float);
  // model->batchnorm1_running_var = NN_tensor(1, (size_t[]){2, }, DTYPE_F32, array_pointer);
  model->batchnorm1_running_var = NN_zeros(1, (size_t[]){2, }, DTYPE_F32);
  array_pointer += 2 * sizeof(float);
  model->batchnorm1_out = NN_tensor(4, (size_t[]){1, 2, 4, 4}, DTYPE_F32, NULL);
}

/**
 * Forward pass of the model
 */
void forward(Model *model) {
  NN_Conv2d_F32(
    model->conv1_out, model->input,
    model->conv1_weight, NULL,
    (size_t[]){3, 3}, (size_t[]){1, 1}, (size_t[]){1, 1}, 1
    );
  NN_BatchNorm2d_F32(
    model->batchnorm1_out, model->conv1_out,
    model->batchnorm1_weight, model->batchnorm1_bias,
    1e-5, 0.1, model->batchnorm1_running_mean, model->batchnorm1_running_var
    );
}

int main() {
  size_t size = (size_t)weights_end - (size_t)weights_start;
  printf("weight size: %d\n", (int)size);

  Model *model = malloc(sizeof(Model));
  
  init(model);

  forward(model);
  
  printf("input:\n");
  NN_printShape(model->input);
  printf("\n");
  NN_printf(model->input);

  printf("output:\n");
  NN_printShape(model->conv1_out);
  printf("\n");
  NN_printf(model->batchnorm1_out);

  return 0;
}
