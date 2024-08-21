#ifndef __MODEL_H
#define __MODEL_H

#include <stdint.h>
#include <stddef.h>
#include "nn.h"


// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "../model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {
  Tensor input_1;
  Tensor seq_0_weight;
  Tensor seq_0_bias;
  Tensor seq_0;
  Tensor seq_1;
  Tensor seq_2_weight;
  Tensor seq_2_bias;
  Tensor seq_2;
  Tensor relu;
  Tensor linear_weight;
  Tensor linear_bias;
  Tensor linear;

} Model;


void init(Model *model);

void forward(Model *model);

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  float *weight_ptr = (float *)model_weight_data;

  NN_init_tensor(&model->input_1, 2, (size_t[]){ 1, 48 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: seq_0
  NN_init_tensor(&model->seq_0_weight, 2, (size_t[]){ 128, 48 }, DTYPE_F32, weight_ptr);
  weight_ptr += 6144;
  NN_init_tensor(&model->seq_0_bias, 1, (size_t[]){ 128 }, DTYPE_F32, weight_ptr);
  weight_ptr += 128;
  NN_init_tensor(&model->seq_0, 2, (size_t[]){ 1, 128 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ELU'>: seq_1
  NN_init_tensor(&model->seq_1, 2, (size_t[]){ 1, 128 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: seq_2
  NN_init_tensor(&model->seq_2_weight, 2, (size_t[]){ 5, 128 }, DTYPE_F32, weight_ptr);
  weight_ptr += 640;
  NN_init_tensor(&model->seq_2_bias, 1, (size_t[]){ 5 }, DTYPE_F32, weight_ptr);
  weight_ptr += 5;
  NN_init_tensor(&model->seq_2, 2, (size_t[]){ 1, 5 }, DTYPE_F32, NULL);
  NN_init_tensor(&model->relu, 2, (size_t[]){ 1, 5 }, DTYPE_F32, NULL);
  NN_init_tensor(&model->linear_weight, 2, (size_t[]){ 12, 5 }, DTYPE_F32, weight_ptr);
  weight_ptr += 60;
  NN_init_tensor(&model->linear_bias, 1, (size_t[]){ 12 }, DTYPE_F32, weight_ptr);
  weight_ptr += 12;
  NN_init_tensor(&model->linear, 2, (size_t[]){ 1, 12 }, DTYPE_F32, NULL);

}


/**
 * Forward pass of the model
 */
void forward(Model *model) {
  NN_linear(&model->seq_0, &model->input_1, &model->seq_0_weight, &model->seq_0_bias);
  NN_elu(&model->seq_1, &model->seq_0, 1.0);
  NN_linear(&model->seq_2, &model->seq_1, &model->seq_2_weight, &model->seq_2_bias);
  NN_relu(&model->relu, &model->seq_2);
  NN_linear(&model->linear, &model->relu, &model->linear_weight, &model->linear_bias);

}

#endif