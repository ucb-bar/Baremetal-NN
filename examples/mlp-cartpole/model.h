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
  Tensor input_1;
  Tensor actor_0_weight;
  Tensor actor_0_bias;
  Tensor actor_0;
  Tensor actor_1;
  Tensor actor_2_weight;
  Tensor actor_2_bias;
  Tensor actor_2;
  Tensor actor_3;
  Tensor actor_4_weight;
  Tensor actor_4_bias;
  Tensor actor_4;
  Tensor actor_5;
  Tensor actor_6_weight;
  Tensor actor_6_bias;
  Tensor actor_6;

} Model;


void init(Model *model);

void forward(Model *model);

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  float *weight_ptr = (float *)model_weight_data;

  nn_init_tensor(&model->input_1, 2, (size_t[]){ 1, 48 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: actor_0
  nn_init_tensor(&model->actor_0_weight, 2, (size_t[]){ 512, 48 }, DTYPE_F32, weight_ptr);
  weight_ptr += 24576;
  nn_init_tensor(&model->actor_0_bias, 1, (size_t[]){ 512 }, DTYPE_F32, weight_ptr);
  weight_ptr += 512;
  nn_init_tensor(&model->actor_0, 2, (size_t[]){ 1, 512 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ELU'>: actor_1
  nn_init_tensor(&model->actor_1, 2, (size_t[]){ 1, 512 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: actor_2
  nn_init_tensor(&model->actor_2_weight, 2, (size_t[]){ 256, 512 }, DTYPE_F32, weight_ptr);
  weight_ptr += 131072;
  nn_init_tensor(&model->actor_2_bias, 1, (size_t[]){ 256 }, DTYPE_F32, weight_ptr);
  weight_ptr += 256;
  nn_init_tensor(&model->actor_2, 2, (size_t[]){ 1, 256 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ELU'>: actor_3
  nn_init_tensor(&model->actor_3, 2, (size_t[]){ 1, 256 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: actor_4
  nn_init_tensor(&model->actor_4_weight, 2, (size_t[]){ 128, 256 }, DTYPE_F32, weight_ptr);
  weight_ptr += 32768;
  nn_init_tensor(&model->actor_4_bias, 1, (size_t[]){ 128 }, DTYPE_F32, weight_ptr);
  weight_ptr += 128;
  nn_init_tensor(&model->actor_4, 2, (size_t[]){ 1, 128 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ELU'>: actor_5
  nn_init_tensor(&model->actor_5, 2, (size_t[]){ 1, 128 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: actor_6
  nn_init_tensor(&model->actor_6_weight, 2, (size_t[]){ 12, 128 }, DTYPE_F32, weight_ptr);
  weight_ptr += 1536;
  nn_init_tensor(&model->actor_6_bias, 1, (size_t[]){ 12 }, DTYPE_F32, weight_ptr);
  weight_ptr += 12;
  nn_init_tensor(&model->actor_6, 2, (size_t[]){ 1, 12 }, DTYPE_F32, NULL);

}


/**
 * Forward pass of the model
 */
void forward(Model *model) {
  nn_linear(&model->actor_0, &model->input_1, &model->actor_0_weight, &model->actor_0_bias);
  nn_elu(&model->actor_1, &model->actor_0, 1.0);
  nn_linear(&model->actor_2, &model->actor_1, &model->actor_2_weight, &model->actor_2_bias);
  nn_elu(&model->actor_3, &model->actor_2, 1.0);
  nn_linear(&model->actor_4, &model->actor_3, &model->actor_4_weight, &model->actor_4_bias);
  nn_elu(&model->actor_5, &model->actor_4, 1.0);
  nn_linear(&model->actor_6, &model->actor_5, &model->actor_6_weight, &model->actor_6_bias);

}

#endif