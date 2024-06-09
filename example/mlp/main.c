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
  Tensor *fc1_weight;
  Tensor *fc1_bias;
  Tensor *fc1_out;
  Tensor *fc2_weight;
  Tensor *fc2_bias;
  Tensor *fc2_out;
  Tensor *fc3_weight;
  Tensor *fc3_bias;
  Tensor *output;
} Model;

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  uint8_t *array_pointer = weights_data;
  
  model->input = NN_ones(2, (size_t[]){1, DIM}, DTYPE_F32);

  model->fc1_weight = NN_tensor(2, (size_t[]){DIM, DIM}, DTYPE_F32, array_pointer);
  array_pointer += DIM * DIM * sizeof(float);
  model->fc1_bias = NN_tensor(2, (size_t[]){1, DIM}, DTYPE_F32, array_pointer);
  array_pointer += DIM * sizeof(float);
  model->fc1_out = NN_tensor(2, (size_t[]){1, DIM}, DTYPE_F32, NULL);

  model->fc2_weight = NN_tensor(2, (size_t[]){DIM, DIM}, DTYPE_F32, array_pointer);
  array_pointer += DIM * DIM * sizeof(float);
  model->fc2_bias = NN_tensor(2, (size_t[]){1, DIM}, DTYPE_F32, array_pointer);
  array_pointer += DIM * sizeof(float);
  model->fc2_out = NN_tensor(2, (size_t[]){1, DIM}, DTYPE_F32, NULL);

  model->fc3_weight = NN_tensor(2, (size_t[]){DIM, DIM}, DTYPE_F32, array_pointer);
  array_pointer += DIM * DIM * sizeof(float);
  model->fc3_bias = NN_tensor(2, (size_t[]){1, DIM}, DTYPE_F32, array_pointer);
  array_pointer += DIM * sizeof(float);

  model->output = NN_tensor(2, (size_t[]){1, DIM}, DTYPE_F32, NULL);

  // printf("fc1_weight: \n");
  // NN_printf(model->fc1_weight);
  // printf("fc1_bias: \n");
  // NN_printf(model->fc1_bias);
}

/**
 * Forward pass of the model
 */
void forward(Model *model) {
  NN_Linear_F32(model->fc1_out, model->input, model->fc1_weight, model->fc1_bias);
  NN_ReLU_F32(model->fc1_out, model->fc1_out);
  NN_Linear_F32(model->fc2_out, model->input, model->fc2_weight, model->fc2_bias);
  NN_ReLU_F32(model->fc2_out, model->fc2_out);
  NN_Linear_F32(model->output, model->input, model->fc3_weight, model->fc3_bias);
}

int main() {
  size_t size = (size_t)weights_end - (size_t)weights_start;
  printf("weight size: %d\n", (int)size);

  Model *model = malloc(sizeof(Model));
  
  init(model);

  forward(model);
  
  printf("input:\n");
  NN_printf(model->input);

  printf("output:\n");
  NN_printf(model->output);

  return 0;
}
