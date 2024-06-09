#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"


// extern char _binary_test_start[];
// extern char _binary_test_end[];


INCLUDE_FILE(".rodata", "../hack_policy.bin", externdata);



#define N_OBS  795
#define N_ACS  14

#define FC1_SIZE 512
#define FC2_SIZE 256
#define FC3_SIZE 128



/* Declaration of symbols (any type can be used) */
extern uint8_t externdata[];
extern size_t externdata_start[];
extern size_t externdata_end[];

typedef struct {
  Tensor input;
  Tensor fc1_weight;
  Tensor fc1_bias;
  Tensor fc1_out;
  Tensor fc2_weight;
  Tensor fc2_bias;
  Tensor fc2_out;
  Tensor fc3_weight;
  Tensor fc3_bias;
  Tensor fc3_out;
  Tensor fc4_weight;
  Tensor fc4_bias;
  Tensor output;
} Model;

void init(Model *model) {
  uint8_t *array_pointer = externdata;

  NN_initTensor(&model->input, 2, (size_t[]){ 1, N_OBS }, DTYPE_F32, (float *)malloc(N_OBS * sizeof(float)));

  NN_initTensor(&model->fc1_weight, 2, (size_t[]){ N_OBS, FC1_SIZE }, DTYPE_F32, (float *)(array_pointer));
  array_pointer += N_OBS * FC1_SIZE * sizeof(float);
  NN_initTensor(&model->fc1_bias, 2, (size_t[]){ 1, FC1_SIZE }, DTYPE_F32, (float *)(array_pointer));
  array_pointer += FC1_SIZE * sizeof(float);
  NN_initTensor(&model->fc1_out, 2, (size_t[]){ 1, FC1_SIZE }, DTYPE_F32, (float *)malloc(FC1_SIZE * sizeof(float)));
  
  NN_initTensor(&model->fc2_weight, 2, (size_t[]){ FC1_SIZE, FC2_SIZE }, DTYPE_F32, (float *)(array_pointer));
  array_pointer += FC1_SIZE * FC2_SIZE * sizeof(float);
  NN_initTensor(&model->fc2_bias, 2, (size_t[]){ 1, FC2_SIZE }, DTYPE_F32, (float *)(array_pointer));
  array_pointer += FC2_SIZE * sizeof(float);
  NN_initTensor(&model->fc2_out, 2, (size_t[]){ 1, FC2_SIZE }, DTYPE_F32, (float *)malloc(FC2_SIZE * sizeof(float)));
  
  NN_initTensor(&model->fc3_weight, 2, (size_t[]){ FC2_SIZE, FC3_SIZE }, DTYPE_F32, (float *)(array_pointer));
  array_pointer += FC2_SIZE * FC3_SIZE * sizeof(float);
  NN_initTensor(&model->fc3_bias, 2, (size_t[]){ 1, FC3_SIZE }, DTYPE_F32, (float *)(array_pointer));
  array_pointer += FC3_SIZE * sizeof(float);
  NN_initTensor(&model->fc3_out, 2, (size_t[]){ 1, FC3_SIZE }, DTYPE_F32, (float *)malloc(FC3_SIZE * sizeof(float)));

  NN_initTensor(&model->fc4_weight, 2, (size_t[]){ FC3_SIZE, N_ACS }, DTYPE_F32, (float *)(array_pointer));
  array_pointer += FC3_SIZE * N_ACS * sizeof(float);
  printf("ptr: %d\n", (int)array_pointer - (int)externdata_start);
  NN_initTensor(&model->fc4_bias, 2, (size_t[]){ 1, N_ACS }, DTYPE_F32, (float *)(array_pointer));
  array_pointer += N_ACS * sizeof(float);
  NN_initTensor(&model->output, 2, (size_t[]){ 1, N_ACS }, DTYPE_F32, (float *)malloc(N_ACS * sizeof(float)));

  printf("fc4_bias: \n");
  NN_printf(&model->fc4_bias);
}

void forward(Model *model) {
  NN_linear_F32(&model->fc1_out, &model->input, &model->fc1_weight, &model->fc1_bias);
  NN_elu_F32(&model->fc1_out, &model->fc1_out, 1.0);
  NN_linear_F32(&model->fc2_out, &model->fc1_out, &model->fc2_weight, &model->fc2_bias);
  NN_elu_F32(&model->fc2_out, &model->fc2_out, 1.0);
  NN_linear_F32(&model->fc3_out, &model->fc2_out, &model->fc3_weight, &model->fc3_bias);
  NN_elu_F32(&model->fc3_out, &model->fc3_out, 1.0);
  NN_linear_F32(&model->output, &model->fc3_out, &model->fc4_weight, &model->fc4_bias);
}

int main() {
    size_t size = (size_t)externdata_end - (size_t)externdata_start;

    printf("size: %d\n", (int)size);

    printf("\n");

  Model *model = (Model *)malloc(sizeof(Model));

  init(model);

  float input_data[N_OBS];
  for (int i = 0; i < N_OBS; i++) {
    input_data[i] = 0.1;
  }
  memcpy(model->input.data, input_data, N_OBS * sizeof(float));
  
  forward(model);

  NN_printf(&model->output);
  
  return 0;
}