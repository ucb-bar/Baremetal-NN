#ifndef __MODEL_H
#define __MODEL_H

#include <stdint.h>
#include <stddef.h>
#include "nn.h"


// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "./model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {
  Tensor x;
  Tensor flatten;
  Tensor fc1_weight;
  Tensor fc1_bias;
  Tensor fc1;
  Tensor relu;
  Tensor fc2_weight;
  Tensor fc2_bias;
  Tensor fc2;
  Tensor relu_1;
  Tensor fc3_weight;
  Tensor fc3_bias;
  Tensor fc3;

} Model;


void init(Model *model);

void forward(Model *model);

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  float *weight_ptr = (float *)model_weight_data;

  nn_init_tensor(&model->x, 4, (size_t[]){ 4, 28, 28, 1 }, DTYPE_F32, NULL);
  nn_init_tensor(&model->flatten, 2, (size_t[]){ 4, 784 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: fc1
  nn_init_tensor(&model->fc1_weight, 2, (size_t[]){ 16, 784 }, DTYPE_F32, weight_ptr);
  weight_ptr += 12544;
  nn_init_tensor(&model->fc1_bias, 1, (size_t[]){ 16 }, DTYPE_F32, weight_ptr);
  weight_ptr += 16;
  nn_init_tensor(&model->fc1, 2, (size_t[]){ 4, 16 }, DTYPE_F32, NULL);
  nn_init_tensor(&model->relu, 2, (size_t[]){ 4, 16 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: fc2
  nn_init_tensor(&model->fc2_weight, 2, (size_t[]){ 16, 16 }, DTYPE_F32, weight_ptr);
  weight_ptr += 256;
  nn_init_tensor(&model->fc2_bias, 1, (size_t[]){ 16 }, DTYPE_F32, weight_ptr);
  weight_ptr += 16;
  nn_init_tensor(&model->fc2, 2, (size_t[]){ 4, 16 }, DTYPE_F32, NULL);
  nn_init_tensor(&model->relu_1, 2, (size_t[]){ 4, 16 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: fc3
  nn_init_tensor(&model->fc3_weight, 2, (size_t[]){ 10, 16 }, DTYPE_F32, weight_ptr);
  weight_ptr += 160;
  nn_init_tensor(&model->fc3_bias, 1, (size_t[]){ 10 }, DTYPE_F32, weight_ptr);
  weight_ptr += 10;
  nn_init_tensor(&model->fc3, 2, (size_t[]){ 4, 10 }, DTYPE_F32, NULL);

}


/**
 * Forward pass of the model
 */
void forward(Model *model) {
  nn_linear(&model->fc1, &model->flatten, &model->fc1_weight, &model->fc1_bias);
  nn_relu(&model->relu, &model->fc1);
  nn_linear(&model->fc2, &model->relu, &model->fc2_weight, &model->fc2_bias);
  nn_relu(&model->relu_1, &model->fc2);
  nn_linear(&model->fc3, &model->relu_1, &model->fc3_weight, &model->fc3_bias);

}

#endif