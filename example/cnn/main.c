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

  Tensor *conv0_0_weight;
  Tensor *conv0_0_out;
  Tensor *conv0_1_weight;
  Tensor *conv0_1_bias;
  Tensor *conv0_1_running_mean;
  Tensor *conv0_1_running_var;
  Tensor *conv0_1_out;
  
  Tensor *conv1_0_weight;
  Tensor *conv1_0_out;
  Tensor *conv1_1_weight;
  Tensor *conv1_1_bias;
  Tensor *conv1_1_running_mean;
  Tensor *conv1_1_running_var;
  Tensor *conv1_1_out;
} Model;

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  uint8_t *array_pointer = weights_data;

  model->input = NN_ones(4, (size_t[]){1, 3, 224, 224}, DTYPE_F32);

  model->conv0_0_weight = NN_tensor(4, (size_t[]){16, 3, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 16 * 3 * 3 * 3 * sizeof(float);
  model->conv0_0_out = NN_tensor(4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);

  model->conv0_1_weight = NN_tensor(1, (size_t[]){16, }, DTYPE_F32, array_pointer);
  // model->conv0_1_weight = NN_ones(1, (size_t[]){16, }, DTYPE_F32);
  array_pointer += 16 * sizeof(float);
  model->conv0_1_bias = NN_tensor(1, (size_t[]){16, }, DTYPE_F32, array_pointer);
  // model->conv0_1_bias = NN_zeros(1, (size_t[]){16, }, DTYPE_F32);
  array_pointer += 16 * sizeof(float);
  model->conv0_1_running_mean = NN_tensor(1, (size_t[]){16, }, DTYPE_F32, array_pointer);
  // model->conv0_1_running_mean = NN_ones(1, (size_t[]){16, }, DTYPE_F32);
  array_pointer += 16 * sizeof(float);
  model->conv0_1_running_var = NN_tensor(1, (size_t[]){16, }, DTYPE_F32, array_pointer);
  // model->conv0_1_running_var = NN_zeros(1, (size_t[]){16, }, DTYPE_F32);
  array_pointer += 16 * sizeof(float);
  model->conv0_1_out = NN_tensor(4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);
  
  size_t size = (size_t)weights_end - (size_t)weights_start;
  printf("weight size: %d\n", (int)size);

  assert((size_t)array_pointer - (size_t)weights_data == size);
}

/**
 * Forward pass of the model
 */
void forward(Model *model) {
  NN_Conv2d_F32(
    model->conv0_0_out, model->input,
    model->conv0_0_weight, NULL,
    (size_t[]){3, 3}, (size_t[]){2, 2}, (size_t[]){1, 1}, 1
    );
  NN_BatchNorm2d_F32(
    model->conv0_1_out, model->conv0_0_out,
    model->conv0_1_weight, model->conv0_1_bias,
    1e-5, 0.1, model->conv0_1_running_mean, model->conv0_1_running_var
    );
  NN_ReLU6Inplace_F32(model->conv0_1_out);
}

int main() {
  Model *model = malloc(sizeof(Model));
  
  init(model);

  forward(model);
  
  // printf("input:\n");
  // NN_printShape(model->input);
  // printf("\n");
  // NN_printf(model->input);

  printf("output:\n");
  NN_printf(model->conv0_1_out);

  return 0;
}
