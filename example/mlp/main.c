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

#define DIM   3


// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "../model.bin", weights);
extern uint8_t weights_data[];
extern size_t weights_start[];
extern size_t weights_end[];

typedef struct {
  Tensor *input;
  Tensor *fc1_weight;
  Tensor *fc1_bias;
  Tensor *output;
} Model;

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  uint8_t *array_pointer = weights_data;

  model->input = NN_tensor(2, (size_t[]){1, DIM}, DTYPE_F32, NULL);
  model->fc1_weight = NN_tensor(2, (size_t[]){DIM, DIM}, DTYPE_F32, array_pointer);
  array_pointer += DIM * DIM * sizeof(float);
  model->fc1_bias = NN_tensor(2, (size_t[]){1, DIM}, DTYPE_F32, array_pointer);
  array_pointer += DIM * sizeof(float);

  model->output = NN_tensor(2, (size_t[]){1, DIM}, DTYPE_F32, NULL);

  printf("fc1_weight: \n");
  NN_printf(model->fc1_weight);
  printf("fc1_bias: \n");
  NN_printf(model->fc1_bias);
}

/**
 * Forward pass of the model
 */
void forward(Model *model) {
  NN_linear_F32(model->output, model->input, model->fc1_weight, model->fc1_bias);
  NN_relu_F32(model->output, model->output);
}

int main() {
  size_t size = (size_t)weights_end - (size_t)weights_start;
  printf("weight size: %d\n", (int)size);

  Model *model = malloc(sizeof(Model));
  
  init(model);
  
  ((float *)model->input->data)[0] = 1.;
  ((float *)model->input->data)[1] = 2.;
  ((float *)model->input->data)[2] = 3.;

  forward(model);
  
  printf("input:\n");
  NN_printf(model->input);

  printf("output:\n");
  NN_printf(model->output);

  return 0;
}
