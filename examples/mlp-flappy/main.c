/**
 * @file main.c
 * 
 * Example runtime for the mlp-flappy network.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"
#include "model.h"

int main() {
  Model *model = malloc(sizeof(Model));

  printf("initalizing model...\n");
  model_init(model);

  printf("setting input data...\n");
  for (int i = 0; i < 83; i += 1) {
    model->obs.data[i] = 1.0;
  }
  
  model_forward(model);

  // output: [[ 0.1256, -0.0136, -0.2046,  0.1883, -0.1451]]
  printf("output:\n");
  nn_print_tensor2d_f32(&model->output);
  
  return 0;
}
