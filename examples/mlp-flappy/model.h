#ifndef __MODEL_H
#define __MODEL_H

#include "nn.h"

// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "./model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {
  Tensor2D_F32 obs;
  Tensor2D_F32 value_value_0;
  Tensor2D_F32 value_value_0_weight;
  Tensor1D_F32 value_value_0_bias;
  Tensor2D_F32 value_value_1;
  Tensor2D_F32 value_value_2;
  Tensor2D_F32 value_value_2_weight;
  Tensor1D_F32 value_value_2_bias;
  Tensor2D_F32 value_value_3;
  Tensor2D_F32 linear;
  Tensor2D_F32 action_action_weight;
  Tensor1D_F32 action_action_bias;
  Tensor2D_F32 output;
} Model;

void model_init(Model* model) {
  model->obs.shape[0] = 1;
  model->obs.shape[1] = 83;
  model->obs.data = (float *)malloc(332);
  model->value_value_0.shape[0] = 1;
  model->value_value_0.shape[1] = 512;
  model->value_value_0.data = (float *)malloc(2048);
  model->value_value_0_weight.shape[0] = 512;
  model->value_value_0_weight.shape[1] = 83;
  model->value_value_0_weight.data = (float *)(model_weight_data + 0);
  model->value_value_0_bias.shape[0] = 512;
  model->value_value_0_bias.data = (float *)(model_weight_data + 169984);
  model->value_value_1.shape[0] = 1;
  model->value_value_1.shape[1] = 512;
  model->value_value_1.data = (float *)malloc(2048);
  model->value_value_2.shape[0] = 1;
  model->value_value_2.shape[1] = 256;
  model->value_value_2.data = (float *)malloc(1024);
  model->value_value_2_weight.shape[0] = 256;
  model->value_value_2_weight.shape[1] = 512;
  model->value_value_2_weight.data = (float *)(model_weight_data + 172032);
  model->value_value_2_bias.shape[0] = 256;
  model->value_value_2_bias.data = (float *)(model_weight_data + 696320);
  model->value_value_3.shape[0] = 1;
  model->value_value_3.shape[1] = 256;
  model->value_value_3.data = (float *)malloc(1024);
  model->linear.shape[0] = 1;
  model->linear.shape[1] = 5;
  model->linear.data = (float *)malloc(20);
  model->action_action_weight.shape[0] = 5;
  model->action_action_weight.shape[1] = 256;
  model->action_action_weight.data = (float *)(model_weight_data + 697344);
  model->action_action_bias.shape[0] = 5;
  model->action_action_bias.data = (float *)(model_weight_data + 702464);
  model->output.shape[0] = 1;
  model->output.shape[1] = 5;
  model->output.data = (float *)malloc(20);
}

void model_forward(Model* model) {
  nn_addmm_f32(&model->value_value_0, &model->obs, &model->value_value_0_weight, &model->value_value_0_bias);
  nn_relu2d_f32(&model->value_value_1, &model->value_value_0);
  nn_addmm_f32(&model->value_value_2, &model->value_value_1, &model->value_value_2_weight, &model->value_value_2_bias);
  nn_relu2d_f32(&model->value_value_3, &model->value_value_2);
  nn_addmm_f32(&model->linear, &model->value_value_3, &model->action_action_weight, &model->action_action_bias);
  memcpy(model->output.data, model->linear.data, 20);
}

#endif  // __MODEL_H
