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
  Tensor2D_F32 _0;
  Tensor2D_F32 _0_weight;
  Tensor1D_F32 _0_bias;
  Tensor2D_F32 _1;
  Tensor2D_F32 _2;
  Tensor2D_F32 _2_weight;
  Tensor1D_F32 _2_bias;
  Tensor2D_F32 _3;
  Tensor2D_F32 _4;
  Tensor2D_F32 _4_weight;
  Tensor1D_F32 _4_bias;
  Tensor2D_F32 _5;
  Tensor2D_F32 _6;
  Tensor2D_F32 _6_weight;
  Tensor1D_F32 _6_bias;
  Tensor2D_F32 output;
} Model;

void model_init(Model* model) {
  model->input_1.shape[0] = 1;
  model->input_1.shape[1] = 81;
  model->input_1.data = (float *)malloc(324);
  model->_0.shape[0] = 1;
  model->_0.shape[1] = 256;
  model->_0.data = (float *)malloc(1024);
  model->_0_weight.shape[0] = 256;
  model->_0_weight.shape[1] = 81;
  model->_0_weight.data = (float *)(model_weight_data + 0);
  model->_0_bias.shape[0] = 256;
  model->_0_bias.data = (float *)(model_weight_data + 82944);
  model->_1.shape[0] = 1;
  model->_1.shape[1] = 256;
  model->_1.data = (float *)malloc(1024);
  model->_2.shape[0] = 1;
  model->_2.shape[1] = 128;
  model->_2.data = (float *)malloc(512);
  model->_2_weight.shape[0] = 128;
  model->_2_weight.shape[1] = 256;
  model->_2_weight.data = (float *)(model_weight_data + 83968);
  model->_2_bias.shape[0] = 128;
  model->_2_bias.data = (float *)(model_weight_data + 215040);
  model->_3.shape[0] = 1;
  model->_3.shape[1] = 128;
  model->_3.data = (float *)malloc(512);
  model->_4.shape[0] = 1;
  model->_4.shape[1] = 128;
  model->_4.data = (float *)malloc(512);
  model->_4_weight.shape[0] = 128;
  model->_4_weight.shape[1] = 128;
  model->_4_weight.data = (float *)(model_weight_data + 215552);
  model->_4_bias.shape[0] = 128;
  model->_4_bias.data = (float *)(model_weight_data + 281088);
  model->_5.shape[0] = 1;
  model->_5.shape[1] = 128;
  model->_5.data = (float *)malloc(512);
  model->_6.shape[0] = 1;
  model->_6.shape[1] = 23;
  model->_6.data = (float *)malloc(92);
  model->_6_weight.shape[0] = 23;
  model->_6_weight.shape[1] = 128;
  model->_6_weight.data = (float *)(model_weight_data + 281600);
  model->_6_bias.shape[0] = 23;
  model->_6_bias.data = (float *)(model_weight_data + 293376);
  model->output.shape[0] = 1;
  model->output.shape[1] = 23;
  model->output.data = (float *)malloc(92);
}

void model_forward(Model* model) {
  nn_linear_f32(&model->_0, &model->input_1, &model->_0_weight, &model->_0_bias);
  nn_elu2d_f32(&model->_1, &model->_0, 1.0);
  nn_linear_f32(&model->_2, &model->_1, &model->_2_weight, &model->_2_bias);
  nn_elu2d_f32(&model->_3, &model->_2, 1.0);
  nn_linear_f32(&model->_4, &model->_3, &model->_4_weight, &model->_4_bias);
  nn_elu2d_f32(&model->_5, &model->_4, 1.0);
  nn_linear_f32(&model->_6, &model->_5, &model->_6_weight, &model->_6_bias);
  memcpy(model->output.data, model->_6.data, 92);
}

#endif  // __MODEL_H
