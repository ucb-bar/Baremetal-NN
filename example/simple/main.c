#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

INCLUDE_FILE(".rodata", "../model.bin", weights);

/* Declaration of symbols (any type can be used) */
extern uint8_t weights_data[];
extern size_t weights_start[];
extern size_t weights_end[];


Tensor A;
Tensor B;
Tensor C;
Tensor D;


void init(Tensor *A, Tensor *B, Tensor *C, Tensor *D) {
  NN_initTensor(A, 2, (size_t[]){3, 3}, DTYPE_F32, (float *)malloc(9 * sizeof(float)));
  NN_initTensor(B, 2, (size_t[]){3, 3}, DTYPE_F32, (float *)(weights_data + 3 * sizeof(float)));
  NN_initTensor(C, 2, (size_t[]){3, 3}, DTYPE_F32, (float *)malloc(9 * sizeof(float)));
  NN_initTensor(D, 1, (size_t[]){3}, DTYPE_F32, (float *)(weights_data + 0 * sizeof(float)));
}

void forward(Tensor *C, Tensor *A, Tensor *B, Tensor *D) {
  NN_linear_F32(C, A, B, D);
}

int main() {
  // size_t size = (size_t)weights_end - (size_t)weights_start;
  // printf("size: %d\n", (int)size);
  // printf("data: ");
  // // [ 0.3110076  -0.6943042   0.39190853 -0.25031644]
  // for (int i = 0; i < size; i+=4) {
  //     // printf("%f ", weights_data[i]);
  //     printf("%f ", *(float *)(weights_data + i));
  // }
  // printf("\n");

  
  init(&A, &B, &C, &D);

  float input_data[] = {1., 2., 3.,  1., 2., 3.,  1., 2., 3.,};
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
  
  return 0;
}