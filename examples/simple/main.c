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

// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "../model.bin", weights);
extern uint8_t weights_data[];
extern size_t weights_start[];
extern size_t weights_end[];

#define DIM   3


// Tensors can be defined either globally or locally
Tensor A;
Tensor B;
Tensor C;
Tensor D;

/**
 * Initialize the required tensors for the model
 */
void init(Tensor *A, Tensor *B, Tensor *C, Tensor *D) {
  NN_init_tensor(A, 2, (size_t[]){3, 3}, DTYPE_F32, (float *)malloc(9 * sizeof(float)));
  NN_init_tensor(B, 2, (size_t[]){3, 3}, DTYPE_F32, (float *)(weights_data + 3 * sizeof(float)));
  NN_init_tensor(C, 2, (size_t[]){3, 3}, DTYPE_F32, (float *)malloc(9 * sizeof(float)));
  NN_init_tensor(D, 1, (size_t[]){3}, DTYPE_F32, (float *)(weights_data + 0 * sizeof(float)));
}

/**
 * Deinitialize the tensors used for the model
 */
void deinit(Tensor *A, Tensor *B, Tensor *C, Tensor *D) {
  NN_freeTensor(A);
  NN_freeTensor(B);
  NN_freeTensor(C);
  NN_freeTensor(D);
}

/**
 * Forward pass of the model
 */
void forward(Tensor *C, Tensor *A, Tensor *B, Tensor *D) {
  NN_Linear_F32(C, A, B, D);
}


int main() {  
  init(&A, &B, &C, &D);

  // load the input data to the tensor
  float input_data[] = {
    1., 2., 3.,
    1., 2., 3.,
    1., 2., 3.,
  };
  memcpy(A.data, input_data, 9 * sizeof(float));
  
  forward(&C, &A, &B, &D);

  printf("A:\n");
  NN_printf(&A);

  printf("B:\n");
  NN_printf(&B);
  
  printf("C:\n");
  NN_printf(&C);
  
  printf("D:\n");
  NN_printf(&D);
  
  deinit(&A, &B, &C, &D);

  return 0;
}
