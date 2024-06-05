#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "nn.h"
#include "riscv_vector.h"

#define N_DIMS 2

static void enable_vector_operations() {
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x00000600 | 0x00006000 | 0x00018000;
    asm volatile("csrw mstatus, %0"::"r"(mstatus));
}

bool float_eq(float golden, float actual, float relErr) {
    return (fabs(actual - golden) < relErr) || (fabs((actual - golden) / actual) < relErr);
}

bool compare_2d(float *golden, float *actual, int n, int m) {
    for (int i = 0; i < m * n; ++i)
        if (!float_eq(golden[i], actual[i], 1e-6))
            return false;
    return true;
}

void NN_random(Tensor *t) {
  for (int i = 0; i < t->size; i += 1) {
    ((float *)t->data)[i] = (float)rand() / RAND_MAX; // + (rand() % 10);
  }
}


#define N  4
#define M  6
#define O  3

int main() {
  enable_vector_operations();
  
  uint32_t seed = 0xdeadbeef;
  uint64_t start, total;
  srand(seed);

  Tensor *A = (Tensor *)malloc(sizeof(Tensor));
  Tensor *B = (Tensor *)malloc(sizeof(Tensor));
  Tensor *B_T = (Tensor *)malloc(sizeof(Tensor));
  Tensor *f = (Tensor *)malloc(sizeof(Tensor));

  NN_initTensor(A, 2, (size_t[]){M, O}, DTYPE_F32, (float *)malloc(M * O * sizeof(float)));
  NN_initTensor(B, 2, (size_t[]){O, N}, DTYPE_F32, (float *)malloc(O * N * sizeof(float)));
  NN_initTensor(f, 2, (size_t[]){M, N}, DTYPE_F32, (float *)malloc(M * N * sizeof(float)));

  NN_random(A);
  NN_random(B);
  NN_random(f);

  printf("matmul:         ");
  Tensor *golden = (Tensor *)malloc(sizeof(Tensor));
  NN_initTensor(golden, 2, (size_t[]){M, N}, DTYPE_F32, (float *)malloc(M * N * sizeof(float)));
  Tensor *actual = (Tensor *)malloc(sizeof(Tensor));
  NN_initTensor(actual, 2, (size_t[]){M, N}, DTYPE_F32, (float *)malloc(M * N * sizeof(float)));
  
  NN_matmul_F32(golden, A, B);
  // start = read_cycles();
  // matmul_rvv(A->data, B->data, actual->data, N, M, O);
  NN_matmul_F32_RVV(actual, A, B);
  // total = read_cycles() - start;
  printf("%s (%lu)\n", compare_2d(golden->data, actual->data, N, M) ? "pass" : "fail", total);

  NN_printf(golden);
  NN_printf(actual);


  return 0;
}