#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

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

void NN_matmul_F32_RVV(Tensor *out, Tensor *a, Tensor *b) {
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
  for (int i = 0; i < a->shape[0]; i += 1) {
    for (int j = 0; j < b->shape[1]; j += 1) {
      float *ptr_a = (float *)a->data + i * a->shape[1];
      float *ptr_b = (float *)b->data + j;
      vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
      size_t vl = 0;
      for (int k = a->shape[1]; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
        vl = __riscv_vsetvl_e32m1(k);
        vfloat32m1_t vec_a = __riscv_vlse32_v_f32m1(ptr_a, sizeof(float), vl);
        vfloat32m1_t vec_b = __riscv_vlse32_v_f32m1(ptr_b, b->shape[1]*sizeof(float), vl);
        vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_a, vec_b, vl);
      }
      vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
      float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
      ((float *)out->data)[i * out->shape[1] + j] = sum;
    }
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