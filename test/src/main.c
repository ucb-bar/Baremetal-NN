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


#define M  3
#define N  4
#define O  5

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
    Tensor *H = NN_rand(2, (size_t[]){N, M}, DTYPE_F32);
    Tensor *V = NN_rand(2, (size_t[]){M, 1}, DTYPE_F32);
    Tensor *W = NN_rand(2, (size_t[]){1, M}, DTYPE_F32);
    
    printf("matvec:\t\t");
    Tensor *golden_vec = NN_tensor(2, (size_t[]){N, 1}, DTYPE_F32, NULL);
    Tensor *actual_vec = NN_tensor(2, (size_t[]){N, 1}, DTYPE_F32, NULL);

    NN_matmul_F32(golden_vec, H, V);
    cycles = READ_CSR("mcycle");
    NN_matmul_F32_RVV(actual_vec, H, V);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s (%lu)\n", compare_2d(golden_vec->data, actual_vec->data, N, 1) ? "pass" : "fail", cycles);

    printf("matvec_t:\t");
    NN_transpose_F32(H, H);
    NN_transpose_F32(golden_vec, golden_vec);
    NN_transpose_F32(actual_vec, actual_vec);
    
    NN_matmul_F32(golden_vec, W, H);
    cycles = READ_CSR("mcycle");
    NN_matmul_F32_RVV(actual_vec, W, H);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s (%lu)\n", compare_2d(golden_vec->data, actual_vec->data, N, 1) ? "pass" : "fail", cycles);

    NN_freeTensorData(H);
    NN_deleteTensor(H);
    NN_freeTensorData(V);
    NN_deleteTensor(V);
    NN_freeTensorData(W);
    NN_deleteTensor(W);
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

    NN_freeTensorData(A);
    NN_deleteTensor(A);
  }

  // matmulf
  {
    Tensor *A = NN_rand(2, (size_t[]){M, N}, DTYPE_F32);
    Tensor *C = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);
    Tensor *D = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);
    
    printf("mulf:\t\t");
    NN_mul_F32(C, A, 10.0f);
    cycles = READ_CSR("mcycle");
    NN_mul_F32_RVV(D, A, 10.0f);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s (%lu)\n", compare_2d(C->data, D->data, M, N) ? "pass" : "fail", cycles);

    NN_freeTensorData(A);
    NN_deleteTensor(A);
    NN_freeTensorData(C);
    NN_deleteTensor(C);
    NN_freeTensorData(D);
    NN_deleteTensor(D);
  }

  // matsub
  {
    Tensor *A = NN_rand(2, (size_t[]){M, N}, DTYPE_F32);
    Tensor *B = NN_rand(2, (size_t[]){M, N}, DTYPE_F32);
    Tensor *golden = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);
    Tensor *actual = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);

    printf("matsub:\t\t");
    NN_sub_F32(golden, A, B);
    cycles = READ_CSR("mcycle");
    NN_sub_F32_RVV(actual, A, B);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s (%lu)\n", compare_2d(golden->data, actual->data, N, M) ? "pass" : "fail", cycles);

    NN_freeTensorData(A);
    NN_deleteTensor(A);
    NN_freeTensorData(B);
    NN_deleteTensor(B);

    NN_freeTensorData(golden);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

  // matadd
  {
    Tensor *A = NN_rand(2, (size_t[]){M, N}, DTYPE_F32);
    Tensor *B = NN_rand(2, (size_t[]){M, N}, DTYPE_F32);
    Tensor *golden = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);
    Tensor *actual = NN_tensor(2, (size_t[]){M, N}, DTYPE_F32, NULL);

    printf("matadd:\t\t");
    NN_add_F32(golden, A, B);
    cycles = READ_CSR("mcycle");
    NN_add_F32_RVV(actual, A, B);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s (%lu)\n", compare_2d(golden->data, actual->data, N, M) ? "pass" : "fail", cycles);

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


  // linear
  {
    int batch = 1;
    int out_features = 4;
    int in_features = 3;

    float weights[] = {-0.00432252, 0.30971584, -0.47518533, -0.4248946,  -0.22236897,  0.15482073, -0.01143914,  0.45777494, -0.0512364,   0.15277413, -0.1744828,  -0.11348708};
    float bias[] = {-0.5515707,  -0.38236874, -0.23799711,  0.02138712};


    Tensor *x = NN_ones(2, (size_t[]){batch, in_features}, DTYPE_F32);
    Tensor *w = NN_tensor(2, (size_t[]){out_features, in_features}, DTYPE_F32, weights);
    Tensor *b = NN_tensor(2, (size_t[]){1, out_features}, DTYPE_F32, bias);

    Tensor *y = NN_tensor(2, (size_t[]){batch, out_features}, DTYPE_F32, NULL);

    // NN_linear_F32(y, x, w, b);
    NN_linear_F32_RVV(y, x, w, b);

    printf("linear:\t\t");
    printf("weights:\n");
    NN_printf(w);
    printf("bias:\n");
    NN_printf(b);
    printf("input:\n");
    NN_printf(x);
    printf("output:\n");
    NN_printf(y);
  }



  // sum
  // {
  //   Tensor *A = NN_ones(2, (size_t[]){M, N}, DTYPE_F32);
  //   Tensor *B = NN_ones(2, (size_t[]){M, 1}, DTYPE_F32);

  //   ((float *)A->data)[0] = 0;
  //   ((float *)A->data)[1] = 1;
  //   ((float *)A->data)[2] = 2;
  //   ((float *)A->data)[3] = 3;
  //   ((float *)A->data)[4] = 4;
  //   ((float *)A->data)[5] = 5;

  //   printf("shape: (%d, %d)\n", A->shape[0], A->shape[1]);
  //   printf("strides: (%d, %d)\n", A->strides[0], A->strides[1]);
    
  //   printf("shape: (%d, %d)\n", B->shape[0], B->shape[1]);
  //   printf("strides: (%d, %d)\n", B->strides[0], B->strides[1]);



  //   // NN_printf(A);

  //   // transpose
  //   size_t shape_0 = A->shape[0];
  //   size_t shape_1 = A->shape[1];
  //   size_t strides_0 = A->strides[0];
  //   size_t strides_1 = A->strides[1];
  //   A->shape[0] = shape_1;
  //   A->shape[1] = shape_0;
  //   A->strides[0] = strides_1;
  //   A->strides[1] = strides_0;
    
  //   printf("shape: (%d, %d)\n", A->shape[0], A->shape[1]);
  //   printf("strides: (%d, %d)\n", A->strides[0], A->strides[1]);

  //   // NN_printf(A);

  //   printf("sum:\t\t");
  //   float sum_cpu = NN_sum_F32(A);
  //   NN_printFloat(sum_cpu, 4);
  //   printf("\n");
    
  // }



  return 0;
}