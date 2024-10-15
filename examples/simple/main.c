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
INCLUDE_FILE(".rodata", "./model.bin", weights);
extern uint8_t weights_data[];
extern size_t weights_start[];
extern size_t weights_end[];

#define DIM   3


// Tensors can be defined either globally or locally
Tensor2D_F32 A;
Tensor2D_F32 B;
Tensor2D_F32 C;
Tensor1D_F32 D;

/**
 * Initialize the required tensors for the model
 */
void init(Tensor2D_F32 *A, Tensor2D_F32 *B, Tensor2D_F32 *C, Tensor1D_F32 *D) {
  A->shape[0] = 3;  A->shape[1] = 3;
  A->data = (float *)malloc(9 * sizeof(float));
  B->shape[0] = 3;  B->shape[1] = 3;
  B->data = (float *)(weights_data + 3 * sizeof(float));
  C->shape[0] = 3;  C->shape[1] = 3;
  C->data = (float *)malloc(9 * sizeof(float));
  D->shape[0] = 3;
  D->data = (float *)(weights_data + 0 * sizeof(float));
}

/**
 * Deinitialize the tensors used for the model
 */
void deinit(Tensor2D_F32 *A, Tensor2D_F32 *B, Tensor2D_F32 *C, Tensor1D_F32 *D) {
  free(A->data);
  free(C->data);
}

/**
 * Forward pass of the model
 */
void forward(Tensor2D_F32 *C, Tensor2D_F32 *A, Tensor2D_F32 *B, Tensor1D_F32 *D) {
  nn_addmm_f32(C, A, B, D);
}


int main() {  
  init(&A, &B, &C, &D);

  // load the input data to the tensor
  float input_data[] = {
    1.0, 2.0, 3.0,
    1.0, 2.0, 3.0,
    1.0, 2.0, 3.0,
  };
  memcpy(A.data, input_data, 9 * sizeof(float));
  
  forward(&C, &A, &B, &D);

  printf("A:\n");
  nn_print_tensor2d_f32(&A);

  printf("B:\n");
  nn_print_tensor2d_f32(&B);
  
  printf("C:\n");
  nn_print_tensor2d_f32(&C);
  
  printf("D:\n");
  nn_print_tensor1d_f32(&D);
  
  deinit(&A, &B, &C, &D);

  return 0;
}
