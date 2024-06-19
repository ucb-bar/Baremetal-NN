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
  init(model);

  printf("setting input data...\n");
  NN_fill(&model->input_1, 0.0);
  
  // cycles = READ_CSR("mcycle");
  forward(model);
  // cycles = READ_CSR("mcycle") - cycles;

  printf("cycles: %lu\n", cycles);

  printf("output:\n");
  NN_printf(&model->actor_2);
  
  return 0;
}
