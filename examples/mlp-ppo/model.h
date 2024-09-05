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
  Tensor _0_weight;
  Tensor _0_bias;
  Tensor _0;
  Tensor _1;
  Tensor _2_weight;
  Tensor _2_bias;
  Tensor _2;
  Tensor _3;
  Tensor _4_weight;
  Tensor _4_bias;
  Tensor _4;
  Tensor _5;
  Tensor _6_weight;
  Tensor _6_bias;
  Tensor _6;

} Model;


void init(Model *model);

void forward(Model *model);

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  float *weight_ptr = (float *)model_weight_data;

  NN_init_tensor(&model->input_1, 2, (size_t[]){ 1, 123 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: _0
  NN_init_tensor(&model->_0_weight, 2, (size_t[]){ 256, 123 }, DTYPE_F32, weight_ptr);
  weight_ptr += 31488;
  NN_init_tensor(&model->_0_bias, 1, (size_t[]){ 256 }, DTYPE_F32, weight_ptr);
  weight_ptr += 256;
  NN_init_tensor(&model->_0, 2, (size_t[]){ 1, 256 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ELU'>: _1
  NN_init_tensor(&model->_1, 2, (size_t[]){ 1, 256 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: _2
  NN_init_tensor(&model->_2_weight, 2, (size_t[]){ 128, 256 }, DTYPE_F32, weight_ptr);
  weight_ptr += 32768;
  NN_init_tensor(&model->_2_bias, 1, (size_t[]){ 128 }, DTYPE_F32, weight_ptr);
  weight_ptr += 128;
  NN_init_tensor(&model->_2, 2, (size_t[]){ 1, 128 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ELU'>: _3
  NN_init_tensor(&model->_3, 2, (size_t[]){ 1, 128 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: _4
  NN_init_tensor(&model->_4_weight, 2, (size_t[]){ 128, 128 }, DTYPE_F32, weight_ptr);
  weight_ptr += 16384;
  NN_init_tensor(&model->_4_bias, 1, (size_t[]){ 128 }, DTYPE_F32, weight_ptr);
  weight_ptr += 128;
  NN_init_tensor(&model->_4, 2, (size_t[]){ 1, 128 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ELU'>: _5
  NN_init_tensor(&model->_5, 2, (size_t[]){ 1, 128 }, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.linear.Linear'>: _6
  NN_init_tensor(&model->_6_weight, 2, (size_t[]){ 37, 128 }, DTYPE_F32, weight_ptr);
  weight_ptr += 4736;
  NN_init_tensor(&model->_6_bias, 1, (size_t[]){ 37 }, DTYPE_F32, weight_ptr);
  weight_ptr += 37;
  NN_init_tensor(&model->_6, 2, (size_t[]){ 1, 37 }, DTYPE_F32, NULL);

}


/**
 * Forward pass of the model
 */
void forward(Model *model) {
  NN_linear(&model->_0, &model->input_1, &model->_0_weight, &model->_0_bias);
  NN_elu(&model->_1, &model->_0, 1.0);
  NN_linear(&model->_2, &model->_1, &model->_2_weight, &model->_2_bias);
  NN_elu(&model->_3, &model->_2, 1.0);
  NN_linear(&model->_4, &model->_3, &model->_4_weight, &model->_4_bias);
  NN_elu(&model->_5, &model->_4, 1.0);
  NN_linear(&model->_6, &model->_5, &model->_6_weight, &model->_6_bias);

}

#endif