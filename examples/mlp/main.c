/**
 * @file main.c
 * 
 * A simple example demonstrating C = A * B + D
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "riscv.h"
#include "nn.h"
#include "model.h"


// static void enable_vector_operations() {
//     unsigned long mstatus;
//     asm volatile("csrr %0, mstatus" : "=r"(mstatus));
//     mstatus |= 0x00000600 | 0x00006000 | 0x00018000;
//     asm volatile("csrw mstatus, %0"::"r"(mstatus));
// }

int main() {

  // enable_vector_operations();
  
  Model *model = malloc(sizeof(Model));

  size_t cycles;
  
  printf("initalizing model...\n");
  model_init(model);

  printf("setting input data...\n");
  for (int i = 0; i < 48; i += 1) {
    model->input_1.data[i] = 1.0;
  }
  
  cycles = READ_CSR("mcycle");
  model_forward(model);
  cycles = READ_CSR("mcycle") - cycles;

  printf("cycles: %lu\n", cycles);

  // output tensor([[ 0.0258, -0.0050,  0.0902, -0.0022, -0.0924, -0.0574,  0.0328,  0.0386, -0.0277,  0.0788,  0.0603, -0.0085]])

  printf("output:\n");
  nn_print_tensor2d_f32(&model->output);
  
  return 0;
}
