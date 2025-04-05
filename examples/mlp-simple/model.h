#ifndef __MODEL_H
#define __MODEL_H

#include "nn.h"

// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "./model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {
  Tensor2D_F32 input_1;
  Tensor2D_F32 actor_0;
  Tensor2D_F32 actor_0_weight;
  Tensor1D_F32 actor_0_bias;
  Tensor2D_F32 actor_1;
  Tensor2D_F32 actor_2;
  Tensor2D_F32 actor_2_weight;
  Tensor1D_F32 actor_2_bias;
  Tensor2D_F32 actor_3;
  Tensor2D_F32 actor_4;
  Tensor2D_F32 actor_4_weight;
  Tensor1D_F32 actor_4_bias;
  Tensor2D_F32 actor_5;
  Tensor2D_F32 actor_6;
  Tensor2D_F32 actor_6_weight;
  Tensor1D_F32 actor_6_bias;
  Tensor2D_F32 output;
} Model;

void model_init(Model* model) {
  model->input_1.shape[0] = 1;
  model->input_1.shape[1] = 48;
  model->input_1.data = (float *)malloc(192);
  model->actor_0.shape[0] = 1;
  model->actor_0.shape[1] = 512;
  model->actor_0.data = (float *)malloc(2048);
  model->actor_0_weight.shape[0] = 512;
  model->actor_0_weight.shape[1] = 48;
  model->actor_0_weight.data = (float *)(model_weight_data + 0);
  model->actor_0_bias.shape[0] = 512;
  model->actor_0_bias.data = (float *)(model_weight_data + 98304);
  model->actor_1.shape[0] = 1;
  model->actor_1.shape[1] = 512;
  model->actor_1.data = (float *)malloc(2048);
  model->actor_2.shape[0] = 1;
  model->actor_2.shape[1] = 256;
  model->actor_2.data = (float *)malloc(1024);
  model->actor_2_weight.shape[0] = 256;
  model->actor_2_weight.shape[1] = 512;
  model->actor_2_weight.data = (float *)(model_weight_data + 100352);
  model->actor_2_bias.shape[0] = 256;
  model->actor_2_bias.data = (float *)(model_weight_data + 624640);
  model->actor_3.shape[0] = 1;
  model->actor_3.shape[1] = 256;
  model->actor_3.data = (float *)malloc(1024);
  model->actor_4.shape[0] = 1;
  model->actor_4.shape[1] = 128;
  model->actor_4.data = (float *)malloc(512);
  model->actor_4_weight.shape[0] = 128;
  model->actor_4_weight.shape[1] = 256;
  model->actor_4_weight.data = (float *)(model_weight_data + 625664);
  model->actor_4_bias.shape[0] = 128;
  model->actor_4_bias.data = (float *)(model_weight_data + 756736);
  model->actor_5.shape[0] = 1;
  model->actor_5.shape[1] = 128;
  model->actor_5.data = (float *)malloc(512);
  model->actor_6.shape[0] = 1;
  model->actor_6.shape[1] = 12;
  model->actor_6.data = (float *)malloc(48);
  model->actor_6_weight.shape[0] = 12;
  model->actor_6_weight.shape[1] = 128;
  model->actor_6_weight.data = (float *)(model_weight_data + 757248);
  model->actor_6_bias.shape[0] = 12;
  model->actor_6_bias.data = (float *)(model_weight_data + 763392);
  model->output.shape[0] = 1;
  model->output.shape[1] = 12;
  model->output.data = (float *)malloc(48);
}

void model_forward(Model* model) {
  nn_linear_f32(&model->actor_0, &model->input_1, &model->actor_0_weight, &model->actor_0_bias);
  nn_elu2d_f32(&model->actor_1, &model->actor_0, 1.0);
  nn_linear_f32(&model->actor_2, &model->actor_1, &model->actor_2_weight, &model->actor_2_bias);
  nn_elu2d_f32(&model->actor_3, &model->actor_2, 1.0);
  nn_linear_f32(&model->actor_4, &model->actor_3, &model->actor_4_weight, &model->actor_4_bias);
  nn_elu2d_f32(&model->actor_5, &model->actor_4, 1.0);
  nn_linear_f32(&model->actor_6, &model->actor_5, &model->actor_6_weight, &model->actor_6_bias);
  memcpy(model->output.data, model->actor_6.data, 48);
}

#endif  // __MODEL_H
