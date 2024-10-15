
#ifndef __MODEL_H
#define __MODEL_H

#include "nn.h"

// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "./model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {
  Tensor2D_F32 seq_0_weight;
  Tensor1D_F32 seq_0_bias;
  Tensor2D_F32 seq_2_weight;
  Tensor1D_F32 seq_2_bias;
  Tensor2D_F32 lin2_weight;
  Tensor1D_F32 lin2_bias;
  Tensor2D_F32 input_1;
  Tensor2D_F32 seq_0;
  Tensor2D_F32 seq_1;
  Tensor2D_F32 seq_2;
  Tensor2D_F32 relu;
  Tensor2D_F32 linear;
  Tensor2D_F32 relu_1;
  Tensor2D_F32 output;
} Model;

void model_init(Model* model) {
  model->seq_0_weight.shape[0] = 128;
  model->seq_0_weight.shape[1] = 48;
  model->seq_0_weight.data = (float *)(model_weight_data + 0);
  model->seq_0_bias.shape[0] = 128;
  model->seq_0_bias.data = (float *)(model_weight_data + 24576);
  model->seq_2_weight.shape[0] = 5;
  model->seq_2_weight.shape[1] = 128;
  model->seq_2_weight.data = (float *)(model_weight_data + 25088);
  model->seq_2_bias.shape[0] = 5;
  model->seq_2_bias.data = (float *)(model_weight_data + 27648);
  model->lin2_weight.shape[0] = 12;
  model->lin2_weight.shape[1] = 5;
  model->lin2_weight.data = (float *)(model_weight_data + 27668);
  model->lin2_bias.shape[0] = 12;
  model->lin2_bias.data = (float *)(model_weight_data + 27908);
  model->input_1.shape[0] = 1;
  model->input_1.shape[1] = 48;
  model->input_1.data = (float *)malloc(192);
  model->seq_0.shape[0] = 1;
  model->seq_0.shape[1] = 128;
  model->seq_0.data = (float *)malloc(512);
  model->seq_1.shape[0] = 1;
  model->seq_1.shape[1] = 128;
  model->seq_1.data = (float *)malloc(512);
  model->seq_2.shape[0] = 1;
  model->seq_2.shape[1] = 5;
  model->seq_2.data = (float *)malloc(20);
  model->relu.shape[0] = 1;
  model->relu.shape[1] = 5;
  model->relu.data = (float *)malloc(20);
  model->linear.shape[0] = 1;
  model->linear.shape[1] = 12;
  model->linear.data = (float *)malloc(48);
  model->relu_1.shape[0] = 1;
  model->relu_1.shape[1] = 5;
  model->relu_1.data = (float *)malloc(20);
  model->output.shape[0] = 1;
  model->output.shape[1] = 12;
  model->output.data = (float *)malloc(48);
}

void model_forward(Model* model) {
  nn_addmm_f32(&model->seq_0, &model->input_1, &model->seq_0_weight, &model->seq_0_bias);
  nn_elu2d_f32(&model->seq_1, &model->seq_0, 1.0);
  nn_addmm_f32(&model->seq_2, &model->seq_1, &model->seq_2_weight, &model->seq_2_bias);
  nn_relu2d_f32(&model->relu, &model->seq_2);
  nn_addmm_f32(&model->linear, &model->relu, &model->lin2_weight, &model->lin2_bias);
  nn_relu2d_f32(&model->relu_1, &model->relu);
  memcpy(model->output.data, model->linear.data, 48);
}

#endif  // __MODEL_H
