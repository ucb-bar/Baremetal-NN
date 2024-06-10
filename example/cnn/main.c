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
#include "model.h"


// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "../input.bin", model_input);
extern uint8_t model_input_data[];
extern size_t model_input_start[];
extern size_t model_input_end[];

int main() {
  Model *model = malloc(sizeof(Model));
  
  init(model);

  // NN_fill_F32(&model->x, 0.0);
  memcpy((uint8_t *)model->x.data, (uint8_t *)model_input_data, (size_t)model_input_end - (size_t)model_input_start);

  // NN_printf(&model->x);

  forward(model);
  
  printf("output:\n");
  NN_printf(&model->decode_conv6_2);

  return 0;
}
