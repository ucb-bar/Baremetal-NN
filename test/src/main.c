#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>

#include "nn.h"
#include "riscv_vector.h"

#define N_DIMS 2

static void enable_vector_operations() {
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x00000600 | 0x00006000 | 0x00018000;
    asm volatile("csrw mstatus, %0"::"r"(mstatus));
}

uint8_t float_eq(float golden, float actual, float relErr) {
  return (fabs(actual - golden) < relErr) || (fabs((actual - golden) / actual) < relErr);
}

uint8_t compare_2d(float *golden, float *actual, int n, int m) {
  for (int i = 0; i < m * n; i+=1) {
    if (!float_eq(golden[i], actual[i], 1e-6)) {
      return 0;
    }
  }
  return 1;
}


#define N  4
#define M  6
#define O  3

int main() {
  enable_vector_operations();
  
  uint32_t seed = 0xdeadbeef;
  size_t cycles;
  srand(seed);

  // matmul
  {
    Tensor *A = NN_rand(2, (size_t[]){M, O}, DTYPE_F32);
    Tensor *B = NN_rand(2, (size_t[]){O, N}, DTYPE_F32);
    Tensor *f = NN_rand(2, (size_t[]){M, N}, DTYPE_F32);

    printf("matmul:\t\t");
    Tensor *golden = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);
    Tensor *actual = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);
    
    NN_matmul_F32(golden, A, B);
    cycles = READ_CSR("mcycle");
    NN_matmul_F32_RVV(actual, A, B);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s (%lu)\n", compare_2d(golden->data, actual->data, N, M) ? "pass" : "fail", cycles);

    // NN_printf(golden);
    // NN_printf(actual);

    NN_freeTensorData(A);
    NN_deleteTensor(A);
    NN_freeTensorData(B);
    NN_deleteTensor(B);
    NN_freeTensorData(f);
    NN_deleteTensor(f);

    NN_freeTensorData(golden);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

  // matvec
  {


  }

  // max and min
  {
    Tensor *A = NN_rand(2, (size_t[]){M, N}, DTYPE_F32);
    
    printf("max:\t\t");
    float max_cpu = NN_max_F32(A);
    cycles = READ_CSR("mcycle");
    float max_actual = NN_max_F32_RVV(A);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s (%lu)\n", float_eq(max_cpu, max_actual, 1e-6) ? "pass" : "fail", cycles);

    printf("min:\t\t");
    float min_cpu = NN_min_F32(A);
    cycles = READ_CSR("mcycle");
    float min_actual =  NN_min_F32_RVV(A);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s (%lu)\n", float_eq(min_cpu, min_actual, 1e-6) ? "pass" : "fail", cycles);

    NN_printf(A);
    printf("max:");
    NN_printFloat(max_cpu, 6);
    printf("\n");

    NN_freeTensorData(A);
    NN_deleteTensor(A);
  }

  // matmulf
  {

  }

  // matsub
  {

  }

  // matadd
  {
    Tensor *A = NN_rand(2, (size_t[]){M, N}, DTYPE_F32);
    Tensor *B = NN_rand(2, (size_t[]){M, N}, DTYPE_F32);
    Tensor *golden = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);
    Tensor *actual = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);

    printf("matadd:\t\t");
    NN_add_F32(golden, A, B);
    // start = read_cycles();
    NN_add_F32_RVV(actual, A, B);
    // total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(golden->data, actual->data, N, M) ? "pass" : "fail", cycles);

    // NN_printf(A);
    // NN_printf(B);
    // NN_printf(golden);
    // NN_printf(actual);

    NN_freeTensorData(A);
    NN_deleteTensor(A);
    NN_freeTensorData(B);
    NN_deleteTensor(B);

    NN_freeTensorData(golden);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

  // matneg
  {

  }

  // matcopy
  {

  }

  // cwiseabs
  {

  }

  // cwisemin
  {

  }

  // cwisemax
  {

  }

  // cwisemul
  {

  }

  // matset
  {

  }

  // matsetv
  {

  }

  // matnorm
  {

  }

  // transpose
  {

  }


  return 0;
}