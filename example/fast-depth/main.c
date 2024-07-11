/**
 * @file main.c
 * 
 * A simple example demonstrating C = A * B + D
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rv.h"
#include "nn.h"
#include "model.h"

#include "termimg.h"

// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "../input.bin", model_input);
extern uint8_t model_input_data[];
extern size_t model_input_start[];
extern size_t model_input_end[];


int main() {
  #ifdef RVV
    printf("Using RVV\n");

    // enable vector instructions
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x00000600 | 0x00006000 | 0x00018000;
    asm volatile("csrw mstatus, %0"::"r"(mstatus));
  #endif
  
  #ifdef GEMMINI
    printf("Using Gemmini\n");
  #endif

  
  Model *model = malloc(sizeof(Model));

  size_t cycles;
  
  printf("initalizing model...\n");
  init(model);

  printf("setting input data...\n");
  // NN_fill(&model->x, 0.0);
  memcpy((uint8_t *)model->x.data, (uint8_t *)model_input_data, (size_t)model_input_end - (size_t)model_input_start);

  // cycles = READ_CSR("mcycle");
  forward(model);
  // cycles = READ_CSR("mcycle") - cycles;

  printf("cycles: %lu\n", cycles);

  Tensor *img = NN_tensor(4, (const size_t[]){1, model->decode_conv6_2.shape[1] / 8, model->decode_conv6_2.shape[2] / 4, 1}, DTYPE_F32, NULL);

  NN_interpolate(img, &model->decode_conv6_2, (float []){0.125, 0.25});

  printf("output:\n");
  show_ASCII_image(img, 0, 0);

  return 0;
}
