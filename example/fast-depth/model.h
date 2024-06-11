#ifndef __MODEL_H
#define __MODEL_H

#include "nn.h"


// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "../model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {
  Tensor x;
  Tensor conv0_0_weight;
  Tensor conv0_0;
  Tensor conv0_1_weight;
  Tensor conv0_1_bias;
  Tensor conv0_1_running_mean;
  Tensor conv0_1_running_var;
  Tensor conv0_1;
  Tensor conv0_2;
  Tensor conv1_0_weight;
  Tensor conv1_0;
  Tensor conv1_1_weight;
  Tensor conv1_1_bias;
  Tensor conv1_1_running_mean;
  Tensor conv1_1_running_var;
  Tensor conv1_1;
  Tensor conv1_2;
  Tensor conv1_3_weight;
  Tensor conv1_3;
  Tensor conv1_4_weight;
  Tensor conv1_4_bias;
  Tensor conv1_4_running_mean;
  Tensor conv1_4_running_var;
  Tensor conv1_4;
  Tensor conv1_5;
  Tensor conv2_0_weight;
  Tensor conv2_0;
  Tensor conv2_1_weight;
  Tensor conv2_1_bias;
  Tensor conv2_1_running_mean;
  Tensor conv2_1_running_var;
  Tensor conv2_1;
  Tensor conv2_2;
  Tensor conv2_3_weight;
  Tensor conv2_3;
  Tensor conv2_4_weight;
  Tensor conv2_4_bias;
  Tensor conv2_4_running_mean;
  Tensor conv2_4_running_var;
  Tensor conv2_4;
  Tensor conv2_5;
  Tensor conv3_0_weight;
  Tensor conv3_0;
  Tensor conv3_1_weight;
  Tensor conv3_1_bias;
  Tensor conv3_1_running_mean;
  Tensor conv3_1_running_var;
  Tensor conv3_1;
  Tensor conv3_2;
  Tensor conv3_3_weight;
  Tensor conv3_3;
  Tensor conv3_4_weight;
  Tensor conv3_4_bias;
  Tensor conv3_4_running_mean;
  Tensor conv3_4_running_var;
  Tensor conv3_4;
  Tensor conv3_5;
  Tensor conv4_0_weight;
  Tensor conv4_0;
  Tensor conv4_1_weight;
  Tensor conv4_1_bias;
  Tensor conv4_1_running_mean;
  Tensor conv4_1_running_var;
  Tensor conv4_1;
  Tensor conv4_2;
  Tensor conv4_3_weight;
  Tensor conv4_3;
  Tensor conv4_4_weight;
  Tensor conv4_4_bias;
  Tensor conv4_4_running_mean;
  Tensor conv4_4_running_var;
  Tensor conv4_4;
  Tensor conv4_5;
  Tensor conv5_0_weight;
  Tensor conv5_0;
  Tensor conv5_1_weight;
  Tensor conv5_1_bias;
  Tensor conv5_1_running_mean;
  Tensor conv5_1_running_var;
  Tensor conv5_1;
  Tensor conv5_2;
  Tensor conv5_3_weight;
  Tensor conv5_3;
  Tensor conv5_4_weight;
  Tensor conv5_4_bias;
  Tensor conv5_4_running_mean;
  Tensor conv5_4_running_var;
  Tensor conv5_4;
  Tensor conv5_5;
  Tensor conv6_0_weight;
  Tensor conv6_0;
  Tensor conv6_1_weight;
  Tensor conv6_1_bias;
  Tensor conv6_1_running_mean;
  Tensor conv6_1_running_var;
  Tensor conv6_1;
  Tensor conv6_2;
  Tensor conv6_3_weight;
  Tensor conv6_3;
  Tensor conv6_4_weight;
  Tensor conv6_4_bias;
  Tensor conv6_4_running_mean;
  Tensor conv6_4_running_var;
  Tensor conv6_4;
  Tensor conv6_5;
  Tensor conv7_0_weight;
  Tensor conv7_0;
  Tensor conv7_1_weight;
  Tensor conv7_1_bias;
  Tensor conv7_1_running_mean;
  Tensor conv7_1_running_var;
  Tensor conv7_1;
  Tensor conv7_2;
  Tensor conv7_3_weight;
  Tensor conv7_3;
  Tensor conv7_4_weight;
  Tensor conv7_4_bias;
  Tensor conv7_4_running_mean;
  Tensor conv7_4_running_var;
  Tensor conv7_4;
  Tensor conv7_5;
  Tensor conv8_0_weight;
  Tensor conv8_0;
  Tensor conv8_1_weight;
  Tensor conv8_1_bias;
  Tensor conv8_1_running_mean;
  Tensor conv8_1_running_var;
  Tensor conv8_1;
  Tensor conv8_2;
  Tensor conv8_3_weight;
  Tensor conv8_3;
  Tensor conv8_4_weight;
  Tensor conv8_4_bias;
  Tensor conv8_4_running_mean;
  Tensor conv8_4_running_var;
  Tensor conv8_4;
  Tensor conv8_5;
  Tensor conv9_0_weight;
  Tensor conv9_0;
  Tensor conv9_1_weight;
  Tensor conv9_1_bias;
  Tensor conv9_1_running_mean;
  Tensor conv9_1_running_var;
  Tensor conv9_1;
  Tensor conv9_2;
  Tensor conv9_3_weight;
  Tensor conv9_3;
  Tensor conv9_4_weight;
  Tensor conv9_4_bias;
  Tensor conv9_4_running_mean;
  Tensor conv9_4_running_var;
  Tensor conv9_4;
  Tensor conv9_5;
  Tensor conv10_0_weight;
  Tensor conv10_0;
  Tensor conv10_1_weight;
  Tensor conv10_1_bias;
  Tensor conv10_1_running_mean;
  Tensor conv10_1_running_var;
  Tensor conv10_1;
  Tensor conv10_2;
  Tensor conv10_3_weight;
  Tensor conv10_3;
  Tensor conv10_4_weight;
  Tensor conv10_4_bias;
  Tensor conv10_4_running_mean;
  Tensor conv10_4_running_var;
  Tensor conv10_4;
  Tensor conv10_5;
  Tensor conv11_0_weight;
  Tensor conv11_0;
  Tensor conv11_1_weight;
  Tensor conv11_1_bias;
  Tensor conv11_1_running_mean;
  Tensor conv11_1_running_var;
  Tensor conv11_1;
  Tensor conv11_2;
  Tensor conv11_3_weight;
  Tensor conv11_3;
  Tensor conv11_4_weight;
  Tensor conv11_4_bias;
  Tensor conv11_4_running_mean;
  Tensor conv11_4_running_var;
  Tensor conv11_4;
  Tensor conv11_5;
  Tensor conv12_0_weight;
  Tensor conv12_0;
  Tensor conv12_1_weight;
  Tensor conv12_1_bias;
  Tensor conv12_1_running_mean;
  Tensor conv12_1_running_var;
  Tensor conv12_1;
  Tensor conv12_2;
  Tensor conv12_3_weight;
  Tensor conv12_3;
  Tensor conv12_4_weight;
  Tensor conv12_4_bias;
  Tensor conv12_4_running_mean;
  Tensor conv12_4_running_var;
  Tensor conv12_4;
  Tensor conv12_5;
  Tensor conv13_0_weight;
  Tensor conv13_0;
  Tensor conv13_1_weight;
  Tensor conv13_1_bias;
  Tensor conv13_1_running_mean;
  Tensor conv13_1_running_var;
  Tensor conv13_1;
  Tensor conv13_2;
  Tensor conv13_3_weight;
  Tensor conv13_3;
  Tensor conv13_4_weight;
  Tensor conv13_4_bias;
  Tensor conv13_4_running_mean;
  Tensor conv13_4_running_var;
  Tensor conv13_4;
  Tensor conv13_5;
  Tensor decode_conv1_0_0_weight;
  Tensor decode_conv1_0_0;
  Tensor decode_conv1_0_1_weight;
  Tensor decode_conv1_0_1_bias;
  Tensor decode_conv1_0_1_running_mean;
  Tensor decode_conv1_0_1_running_var;
  Tensor decode_conv1_0_1;
  Tensor decode_conv1_0_2;
  Tensor decode_conv1_1_0_weight;
  Tensor decode_conv1_1_0;
  Tensor decode_conv1_1_1_weight;
  Tensor decode_conv1_1_1_bias;
  Tensor decode_conv1_1_1_running_mean;
  Tensor decode_conv1_1_1_running_var;
  Tensor decode_conv1_1_1;
  Tensor decode_conv1_1_2;
  Tensor interpolate;
  Tensor decode_conv2_0_0_weight;
  Tensor decode_conv2_0_0;
  Tensor decode_conv2_0_1_weight;
  Tensor decode_conv2_0_1_bias;
  Tensor decode_conv2_0_1_running_mean;
  Tensor decode_conv2_0_1_running_var;
  Tensor decode_conv2_0_1;
  Tensor decode_conv2_0_2;
  Tensor decode_conv2_1_0_weight;
  Tensor decode_conv2_1_0;
  Tensor decode_conv2_1_1_weight;
  Tensor decode_conv2_1_1_bias;
  Tensor decode_conv2_1_1_running_mean;
  Tensor decode_conv2_1_1_running_var;
  Tensor decode_conv2_1_1;
  Tensor decode_conv2_1_2;
  Tensor interpolate_1;
  Tensor add;
  Tensor decode_conv3_0_0_weight;
  Tensor decode_conv3_0_0;
  Tensor decode_conv3_0_1_weight;
  Tensor decode_conv3_0_1_bias;
  Tensor decode_conv3_0_1_running_mean;
  Tensor decode_conv3_0_1_running_var;
  Tensor decode_conv3_0_1;
  Tensor decode_conv3_0_2;
  Tensor decode_conv3_1_0_weight;
  Tensor decode_conv3_1_0;
  Tensor decode_conv3_1_1_weight;
  Tensor decode_conv3_1_1_bias;
  Tensor decode_conv3_1_1_running_mean;
  Tensor decode_conv3_1_1_running_var;
  Tensor decode_conv3_1_1;
  Tensor decode_conv3_1_2;
  Tensor interpolate_2;
  Tensor add_1;
  Tensor decode_conv4_0_0_weight;
  Tensor decode_conv4_0_0;
  Tensor decode_conv4_0_1_weight;
  Tensor decode_conv4_0_1_bias;
  Tensor decode_conv4_0_1_running_mean;
  Tensor decode_conv4_0_1_running_var;
  Tensor decode_conv4_0_1;
  Tensor decode_conv4_0_2;
  Tensor decode_conv4_1_0_weight;
  Tensor decode_conv4_1_0;
  Tensor decode_conv4_1_1_weight;
  Tensor decode_conv4_1_1_bias;
  Tensor decode_conv4_1_1_running_mean;
  Tensor decode_conv4_1_1_running_var;
  Tensor decode_conv4_1_1;
  Tensor decode_conv4_1_2;
  Tensor interpolate_3;
  Tensor add_2;
  Tensor decode_conv5_0_0_weight;
  Tensor decode_conv5_0_0;
  Tensor decode_conv5_0_1_weight;
  Tensor decode_conv5_0_1_bias;
  Tensor decode_conv5_0_1_running_mean;
  Tensor decode_conv5_0_1_running_var;
  Tensor decode_conv5_0_1;
  Tensor decode_conv5_0_2;
  Tensor decode_conv5_1_0_weight;
  Tensor decode_conv5_1_0;
  Tensor decode_conv5_1_1_weight;
  Tensor decode_conv5_1_1_bias;
  Tensor decode_conv5_1_1_running_mean;
  Tensor decode_conv5_1_1_running_var;
  Tensor decode_conv5_1_1;
  Tensor decode_conv5_1_2;
  Tensor interpolate_4;
  Tensor decode_conv6_0_weight;
  Tensor decode_conv6_0;
  Tensor decode_conv6_1_weight;
  Tensor decode_conv6_1_bias;
  Tensor decode_conv6_1_running_mean;
  Tensor decode_conv6_1_running_var;
  Tensor decode_conv6_1;
  Tensor decode_conv6_2;

} Model;


void init(Model *model);

void forward(Model *model);

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  float *array_pointer = (float *)model_weight_data;

  NN_initTensor(&model->x, 4, (size_t[]){1, 3, 224, 224}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv0_0
  NN_initTensor(&model->conv0_0_weight, 4, (size_t[]){16, 3, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 432;
  NN_initTensor(&model->conv0_0, 4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv0_1
  NN_initTensor(&model->conv0_1_weight, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->conv0_1_bias, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->conv0_1_running_mean, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->conv0_1_running_var, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->conv0_1, 4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv0_2
  NN_initTensor(&model->conv0_2, 4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv1_0
  NN_initTensor(&model->conv1_0_weight, 4, (size_t[]){16, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 144;
  NN_initTensor(&model->conv1_0, 4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv1_1
  NN_initTensor(&model->conv1_1_weight, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->conv1_1_bias, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->conv1_1_running_mean, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->conv1_1_running_var, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->conv1_1, 4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv1_2
  NN_initTensor(&model->conv1_2, 4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv1_3
  NN_initTensor(&model->conv1_3_weight, 4, (size_t[]){56, 16, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 896;
  NN_initTensor(&model->conv1_3, 4, (size_t[]){1, 56, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv1_4
  NN_initTensor(&model->conv1_4_weight, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->conv1_4_bias, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->conv1_4_running_mean, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->conv1_4_running_var, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->conv1_4, 4, (size_t[]){1, 56, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv1_5
  NN_initTensor(&model->conv1_5, 4, (size_t[]){1, 56, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv2_0
  NN_initTensor(&model->conv2_0_weight, 4, (size_t[]){56, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 504;
  NN_initTensor(&model->conv2_0, 4, (size_t[]){1, 56, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv2_1
  NN_initTensor(&model->conv2_1_weight, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->conv2_1_bias, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->conv2_1_running_mean, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->conv2_1_running_var, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->conv2_1, 4, (size_t[]){1, 56, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv2_2
  NN_initTensor(&model->conv2_2, 4, (size_t[]){1, 56, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv2_3
  NN_initTensor(&model->conv2_3_weight, 4, (size_t[]){88, 56, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 4928;
  NN_initTensor(&model->conv2_3, 4, (size_t[]){1, 88, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv2_4
  NN_initTensor(&model->conv2_4_weight, 1, (size_t[]){88}, DTYPE_F32, array_pointer);
  array_pointer += 88;
  NN_initTensor(&model->conv2_4_bias, 1, (size_t[]){88}, DTYPE_F32, array_pointer);
  array_pointer += 88;
  NN_initTensor(&model->conv2_4_running_mean, 1, (size_t[]){88}, DTYPE_F32, array_pointer);
  array_pointer += 88;
  NN_initTensor(&model->conv2_4_running_var, 1, (size_t[]){88}, DTYPE_F32, array_pointer);
  array_pointer += 88;
  NN_initTensor(&model->conv2_4, 4, (size_t[]){1, 88, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv2_5
  NN_initTensor(&model->conv2_5, 4, (size_t[]){1, 88, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv3_0
  NN_initTensor(&model->conv3_0_weight, 4, (size_t[]){88, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 792;
  NN_initTensor(&model->conv3_0, 4, (size_t[]){1, 88, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv3_1
  NN_initTensor(&model->conv3_1_weight, 1, (size_t[]){88}, DTYPE_F32, array_pointer);
  array_pointer += 88;
  NN_initTensor(&model->conv3_1_bias, 1, (size_t[]){88}, DTYPE_F32, array_pointer);
  array_pointer += 88;
  NN_initTensor(&model->conv3_1_running_mean, 1, (size_t[]){88}, DTYPE_F32, array_pointer);
  array_pointer += 88;
  NN_initTensor(&model->conv3_1_running_var, 1, (size_t[]){88}, DTYPE_F32, array_pointer);
  array_pointer += 88;
  NN_initTensor(&model->conv3_1, 4, (size_t[]){1, 88, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv3_2
  NN_initTensor(&model->conv3_2, 4, (size_t[]){1, 88, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv3_3
  NN_initTensor(&model->conv3_3_weight, 4, (size_t[]){120, 88, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 10560;
  NN_initTensor(&model->conv3_3, 4, (size_t[]){1, 120, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv3_4
  NN_initTensor(&model->conv3_4_weight, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->conv3_4_bias, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->conv3_4_running_mean, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->conv3_4_running_var, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->conv3_4, 4, (size_t[]){1, 120, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv3_5
  NN_initTensor(&model->conv3_5, 4, (size_t[]){1, 120, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv4_0
  NN_initTensor(&model->conv4_0_weight, 4, (size_t[]){120, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 1080;
  NN_initTensor(&model->conv4_0, 4, (size_t[]){1, 120, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv4_1
  NN_initTensor(&model->conv4_1_weight, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->conv4_1_bias, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->conv4_1_running_mean, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->conv4_1_running_var, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->conv4_1, 4, (size_t[]){1, 120, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv4_2
  NN_initTensor(&model->conv4_2, 4, (size_t[]){1, 120, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv4_3
  NN_initTensor(&model->conv4_3_weight, 4, (size_t[]){144, 120, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 17280;
  NN_initTensor(&model->conv4_3, 4, (size_t[]){1, 144, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv4_4
  NN_initTensor(&model->conv4_4_weight, 1, (size_t[]){144}, DTYPE_F32, array_pointer);
  array_pointer += 144;
  NN_initTensor(&model->conv4_4_bias, 1, (size_t[]){144}, DTYPE_F32, array_pointer);
  array_pointer += 144;
  NN_initTensor(&model->conv4_4_running_mean, 1, (size_t[]){144}, DTYPE_F32, array_pointer);
  array_pointer += 144;
  NN_initTensor(&model->conv4_4_running_var, 1, (size_t[]){144}, DTYPE_F32, array_pointer);
  array_pointer += 144;
  NN_initTensor(&model->conv4_4, 4, (size_t[]){1, 144, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv4_5
  NN_initTensor(&model->conv4_5, 4, (size_t[]){1, 144, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv5_0
  NN_initTensor(&model->conv5_0_weight, 4, (size_t[]){144, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 1296;
  NN_initTensor(&model->conv5_0, 4, (size_t[]){1, 144, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv5_1
  NN_initTensor(&model->conv5_1_weight, 1, (size_t[]){144}, DTYPE_F32, array_pointer);
  array_pointer += 144;
  NN_initTensor(&model->conv5_1_bias, 1, (size_t[]){144}, DTYPE_F32, array_pointer);
  array_pointer += 144;
  NN_initTensor(&model->conv5_1_running_mean, 1, (size_t[]){144}, DTYPE_F32, array_pointer);
  array_pointer += 144;
  NN_initTensor(&model->conv5_1_running_var, 1, (size_t[]){144}, DTYPE_F32, array_pointer);
  array_pointer += 144;
  NN_initTensor(&model->conv5_1, 4, (size_t[]){1, 144, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv5_2
  NN_initTensor(&model->conv5_2, 4, (size_t[]){1, 144, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv5_3
  NN_initTensor(&model->conv5_3_weight, 4, (size_t[]){256, 144, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 36864;
  NN_initTensor(&model->conv5_3, 4, (size_t[]){1, 256, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv5_4
  NN_initTensor(&model->conv5_4_weight, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->conv5_4_bias, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->conv5_4_running_mean, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->conv5_4_running_var, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->conv5_4, 4, (size_t[]){1, 256, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv5_5
  NN_initTensor(&model->conv5_5, 4, (size_t[]){1, 256, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv6_0
  NN_initTensor(&model->conv6_0_weight, 4, (size_t[]){256, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 2304;
  NN_initTensor(&model->conv6_0, 4, (size_t[]){1, 256, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv6_1
  NN_initTensor(&model->conv6_1_weight, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->conv6_1_bias, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->conv6_1_running_mean, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->conv6_1_running_var, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->conv6_1, 4, (size_t[]){1, 256, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv6_2
  NN_initTensor(&model->conv6_2, 4, (size_t[]){1, 256, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv6_3
  NN_initTensor(&model->conv6_3_weight, 4, (size_t[]){408, 256, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 104448;
  NN_initTensor(&model->conv6_3, 4, (size_t[]){1, 408, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv6_4
  NN_initTensor(&model->conv6_4_weight, 1, (size_t[]){408}, DTYPE_F32, array_pointer);
  array_pointer += 408;
  NN_initTensor(&model->conv6_4_bias, 1, (size_t[]){408}, DTYPE_F32, array_pointer);
  array_pointer += 408;
  NN_initTensor(&model->conv6_4_running_mean, 1, (size_t[]){408}, DTYPE_F32, array_pointer);
  array_pointer += 408;
  NN_initTensor(&model->conv6_4_running_var, 1, (size_t[]){408}, DTYPE_F32, array_pointer);
  array_pointer += 408;
  NN_initTensor(&model->conv6_4, 4, (size_t[]){1, 408, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv6_5
  NN_initTensor(&model->conv6_5, 4, (size_t[]){1, 408, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv7_0
  NN_initTensor(&model->conv7_0_weight, 4, (size_t[]){408, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 3672;
  NN_initTensor(&model->conv7_0, 4, (size_t[]){1, 408, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv7_1
  NN_initTensor(&model->conv7_1_weight, 1, (size_t[]){408}, DTYPE_F32, array_pointer);
  array_pointer += 408;
  NN_initTensor(&model->conv7_1_bias, 1, (size_t[]){408}, DTYPE_F32, array_pointer);
  array_pointer += 408;
  NN_initTensor(&model->conv7_1_running_mean, 1, (size_t[]){408}, DTYPE_F32, array_pointer);
  array_pointer += 408;
  NN_initTensor(&model->conv7_1_running_var, 1, (size_t[]){408}, DTYPE_F32, array_pointer);
  array_pointer += 408;
  NN_initTensor(&model->conv7_1, 4, (size_t[]){1, 408, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv7_2
  NN_initTensor(&model->conv7_2, 4, (size_t[]){1, 408, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv7_3
  NN_initTensor(&model->conv7_3_weight, 4, (size_t[]){376, 408, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 153408;
  NN_initTensor(&model->conv7_3, 4, (size_t[]){1, 376, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv7_4
  NN_initTensor(&model->conv7_4_weight, 1, (size_t[]){376}, DTYPE_F32, array_pointer);
  array_pointer += 376;
  NN_initTensor(&model->conv7_4_bias, 1, (size_t[]){376}, DTYPE_F32, array_pointer);
  array_pointer += 376;
  NN_initTensor(&model->conv7_4_running_mean, 1, (size_t[]){376}, DTYPE_F32, array_pointer);
  array_pointer += 376;
  NN_initTensor(&model->conv7_4_running_var, 1, (size_t[]){376}, DTYPE_F32, array_pointer);
  array_pointer += 376;
  NN_initTensor(&model->conv7_4, 4, (size_t[]){1, 376, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv7_5
  NN_initTensor(&model->conv7_5, 4, (size_t[]){1, 376, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv8_0
  NN_initTensor(&model->conv8_0_weight, 4, (size_t[]){376, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 3384;
  NN_initTensor(&model->conv8_0, 4, (size_t[]){1, 376, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv8_1
  NN_initTensor(&model->conv8_1_weight, 1, (size_t[]){376}, DTYPE_F32, array_pointer);
  array_pointer += 376;
  NN_initTensor(&model->conv8_1_bias, 1, (size_t[]){376}, DTYPE_F32, array_pointer);
  array_pointer += 376;
  NN_initTensor(&model->conv8_1_running_mean, 1, (size_t[]){376}, DTYPE_F32, array_pointer);
  array_pointer += 376;
  NN_initTensor(&model->conv8_1_running_var, 1, (size_t[]){376}, DTYPE_F32, array_pointer);
  array_pointer += 376;
  NN_initTensor(&model->conv8_1, 4, (size_t[]){1, 376, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv8_2
  NN_initTensor(&model->conv8_2, 4, (size_t[]){1, 376, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv8_3
  NN_initTensor(&model->conv8_3_weight, 4, (size_t[]){272, 376, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 102272;
  NN_initTensor(&model->conv8_3, 4, (size_t[]){1, 272, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv8_4
  NN_initTensor(&model->conv8_4_weight, 1, (size_t[]){272}, DTYPE_F32, array_pointer);
  array_pointer += 272;
  NN_initTensor(&model->conv8_4_bias, 1, (size_t[]){272}, DTYPE_F32, array_pointer);
  array_pointer += 272;
  NN_initTensor(&model->conv8_4_running_mean, 1, (size_t[]){272}, DTYPE_F32, array_pointer);
  array_pointer += 272;
  NN_initTensor(&model->conv8_4_running_var, 1, (size_t[]){272}, DTYPE_F32, array_pointer);
  array_pointer += 272;
  NN_initTensor(&model->conv8_4, 4, (size_t[]){1, 272, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv8_5
  NN_initTensor(&model->conv8_5, 4, (size_t[]){1, 272, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv9_0
  NN_initTensor(&model->conv9_0_weight, 4, (size_t[]){272, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 2448;
  NN_initTensor(&model->conv9_0, 4, (size_t[]){1, 272, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv9_1
  NN_initTensor(&model->conv9_1_weight, 1, (size_t[]){272}, DTYPE_F32, array_pointer);
  array_pointer += 272;
  NN_initTensor(&model->conv9_1_bias, 1, (size_t[]){272}, DTYPE_F32, array_pointer);
  array_pointer += 272;
  NN_initTensor(&model->conv9_1_running_mean, 1, (size_t[]){272}, DTYPE_F32, array_pointer);
  array_pointer += 272;
  NN_initTensor(&model->conv9_1_running_var, 1, (size_t[]){272}, DTYPE_F32, array_pointer);
  array_pointer += 272;
  NN_initTensor(&model->conv9_1, 4, (size_t[]){1, 272, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv9_2
  NN_initTensor(&model->conv9_2, 4, (size_t[]){1, 272, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv9_3
  NN_initTensor(&model->conv9_3_weight, 4, (size_t[]){288, 272, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 78336;
  NN_initTensor(&model->conv9_3, 4, (size_t[]){1, 288, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv9_4
  NN_initTensor(&model->conv9_4_weight, 1, (size_t[]){288}, DTYPE_F32, array_pointer);
  array_pointer += 288;
  NN_initTensor(&model->conv9_4_bias, 1, (size_t[]){288}, DTYPE_F32, array_pointer);
  array_pointer += 288;
  NN_initTensor(&model->conv9_4_running_mean, 1, (size_t[]){288}, DTYPE_F32, array_pointer);
  array_pointer += 288;
  NN_initTensor(&model->conv9_4_running_var, 1, (size_t[]){288}, DTYPE_F32, array_pointer);
  array_pointer += 288;
  NN_initTensor(&model->conv9_4, 4, (size_t[]){1, 288, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv9_5
  NN_initTensor(&model->conv9_5, 4, (size_t[]){1, 288, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv10_0
  NN_initTensor(&model->conv10_0_weight, 4, (size_t[]){288, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 2592;
  NN_initTensor(&model->conv10_0, 4, (size_t[]){1, 288, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv10_1
  NN_initTensor(&model->conv10_1_weight, 1, (size_t[]){288}, DTYPE_F32, array_pointer);
  array_pointer += 288;
  NN_initTensor(&model->conv10_1_bias, 1, (size_t[]){288}, DTYPE_F32, array_pointer);
  array_pointer += 288;
  NN_initTensor(&model->conv10_1_running_mean, 1, (size_t[]){288}, DTYPE_F32, array_pointer);
  array_pointer += 288;
  NN_initTensor(&model->conv10_1_running_var, 1, (size_t[]){288}, DTYPE_F32, array_pointer);
  array_pointer += 288;
  NN_initTensor(&model->conv10_1, 4, (size_t[]){1, 288, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv10_2
  NN_initTensor(&model->conv10_2, 4, (size_t[]){1, 288, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv10_3
  NN_initTensor(&model->conv10_3_weight, 4, (size_t[]){296, 288, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 85248;
  NN_initTensor(&model->conv10_3, 4, (size_t[]){1, 296, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv10_4
  NN_initTensor(&model->conv10_4_weight, 1, (size_t[]){296}, DTYPE_F32, array_pointer);
  array_pointer += 296;
  NN_initTensor(&model->conv10_4_bias, 1, (size_t[]){296}, DTYPE_F32, array_pointer);
  array_pointer += 296;
  NN_initTensor(&model->conv10_4_running_mean, 1, (size_t[]){296}, DTYPE_F32, array_pointer);
  array_pointer += 296;
  NN_initTensor(&model->conv10_4_running_var, 1, (size_t[]){296}, DTYPE_F32, array_pointer);
  array_pointer += 296;
  NN_initTensor(&model->conv10_4, 4, (size_t[]){1, 296, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv10_5
  NN_initTensor(&model->conv10_5, 4, (size_t[]){1, 296, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv11_0
  NN_initTensor(&model->conv11_0_weight, 4, (size_t[]){296, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 2664;
  NN_initTensor(&model->conv11_0, 4, (size_t[]){1, 296, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv11_1
  NN_initTensor(&model->conv11_1_weight, 1, (size_t[]){296}, DTYPE_F32, array_pointer);
  array_pointer += 296;
  NN_initTensor(&model->conv11_1_bias, 1, (size_t[]){296}, DTYPE_F32, array_pointer);
  array_pointer += 296;
  NN_initTensor(&model->conv11_1_running_mean, 1, (size_t[]){296}, DTYPE_F32, array_pointer);
  array_pointer += 296;
  NN_initTensor(&model->conv11_1_running_var, 1, (size_t[]){296}, DTYPE_F32, array_pointer);
  array_pointer += 296;
  NN_initTensor(&model->conv11_1, 4, (size_t[]){1, 296, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv11_2
  NN_initTensor(&model->conv11_2, 4, (size_t[]){1, 296, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv11_3
  NN_initTensor(&model->conv11_3_weight, 4, (size_t[]){328, 296, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 97088;
  NN_initTensor(&model->conv11_3, 4, (size_t[]){1, 328, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv11_4
  NN_initTensor(&model->conv11_4_weight, 1, (size_t[]){328}, DTYPE_F32, array_pointer);
  array_pointer += 328;
  NN_initTensor(&model->conv11_4_bias, 1, (size_t[]){328}, DTYPE_F32, array_pointer);
  array_pointer += 328;
  NN_initTensor(&model->conv11_4_running_mean, 1, (size_t[]){328}, DTYPE_F32, array_pointer);
  array_pointer += 328;
  NN_initTensor(&model->conv11_4_running_var, 1, (size_t[]){328}, DTYPE_F32, array_pointer);
  array_pointer += 328;
  NN_initTensor(&model->conv11_4, 4, (size_t[]){1, 328, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv11_5
  NN_initTensor(&model->conv11_5, 4, (size_t[]){1, 328, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv12_0
  NN_initTensor(&model->conv12_0_weight, 4, (size_t[]){328, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 2952;
  NN_initTensor(&model->conv12_0, 4, (size_t[]){1, 328, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv12_1
  NN_initTensor(&model->conv12_1_weight, 1, (size_t[]){328}, DTYPE_F32, array_pointer);
  array_pointer += 328;
  NN_initTensor(&model->conv12_1_bias, 1, (size_t[]){328}, DTYPE_F32, array_pointer);
  array_pointer += 328;
  NN_initTensor(&model->conv12_1_running_mean, 1, (size_t[]){328}, DTYPE_F32, array_pointer);
  array_pointer += 328;
  NN_initTensor(&model->conv12_1_running_var, 1, (size_t[]){328}, DTYPE_F32, array_pointer);
  array_pointer += 328;
  NN_initTensor(&model->conv12_1, 4, (size_t[]){1, 328, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv12_2
  NN_initTensor(&model->conv12_2, 4, (size_t[]){1, 328, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv12_3
  NN_initTensor(&model->conv12_3_weight, 4, (size_t[]){480, 328, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 157440;
  NN_initTensor(&model->conv12_3, 4, (size_t[]){1, 480, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv12_4
  NN_initTensor(&model->conv12_4_weight, 1, (size_t[]){480}, DTYPE_F32, array_pointer);
  array_pointer += 480;
  NN_initTensor(&model->conv12_4_bias, 1, (size_t[]){480}, DTYPE_F32, array_pointer);
  array_pointer += 480;
  NN_initTensor(&model->conv12_4_running_mean, 1, (size_t[]){480}, DTYPE_F32, array_pointer);
  array_pointer += 480;
  NN_initTensor(&model->conv12_4_running_var, 1, (size_t[]){480}, DTYPE_F32, array_pointer);
  array_pointer += 480;
  NN_initTensor(&model->conv12_4, 4, (size_t[]){1, 480, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv12_5
  NN_initTensor(&model->conv12_5, 4, (size_t[]){1, 480, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv13_0
  NN_initTensor(&model->conv13_0_weight, 4, (size_t[]){480, 1, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 4320;
  NN_initTensor(&model->conv13_0, 4, (size_t[]){1, 480, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv13_1
  NN_initTensor(&model->conv13_1_weight, 1, (size_t[]){480}, DTYPE_F32, array_pointer);
  array_pointer += 480;
  NN_initTensor(&model->conv13_1_bias, 1, (size_t[]){480}, DTYPE_F32, array_pointer);
  array_pointer += 480;
  NN_initTensor(&model->conv13_1_running_mean, 1, (size_t[]){480}, DTYPE_F32, array_pointer);
  array_pointer += 480;
  NN_initTensor(&model->conv13_1_running_var, 1, (size_t[]){480}, DTYPE_F32, array_pointer);
  array_pointer += 480;
  NN_initTensor(&model->conv13_1, 4, (size_t[]){1, 480, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv13_2
  NN_initTensor(&model->conv13_2, 4, (size_t[]){1, 480, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv13_3
  NN_initTensor(&model->conv13_3_weight, 4, (size_t[]){512, 480, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 245760;
  NN_initTensor(&model->conv13_3, 4, (size_t[]){1, 512, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv13_4
  NN_initTensor(&model->conv13_4_weight, 1, (size_t[]){512}, DTYPE_F32, array_pointer);
  array_pointer += 512;
  NN_initTensor(&model->conv13_4_bias, 1, (size_t[]){512}, DTYPE_F32, array_pointer);
  array_pointer += 512;
  NN_initTensor(&model->conv13_4_running_mean, 1, (size_t[]){512}, DTYPE_F32, array_pointer);
  array_pointer += 512;
  NN_initTensor(&model->conv13_4_running_var, 1, (size_t[]){512}, DTYPE_F32, array_pointer);
  array_pointer += 512;
  NN_initTensor(&model->conv13_4, 4, (size_t[]){1, 512, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv13_5
  NN_initTensor(&model->conv13_5, 4, (size_t[]){1, 512, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv1_0_0
  NN_initTensor(&model->decode_conv1_0_0_weight, 4, (size_t[]){512, 1, 5, 5}, DTYPE_F32, array_pointer);
  array_pointer += 12800;
  NN_initTensor(&model->decode_conv1_0_0, 4, (size_t[]){1, 512, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv1_0_1
  NN_initTensor(&model->decode_conv1_0_1_weight, 1, (size_t[]){512}, DTYPE_F32, array_pointer);
  array_pointer += 512;
  NN_initTensor(&model->decode_conv1_0_1_bias, 1, (size_t[]){512}, DTYPE_F32, array_pointer);
  array_pointer += 512;
  NN_initTensor(&model->decode_conv1_0_1_running_mean, 1, (size_t[]){512}, DTYPE_F32, array_pointer);
  array_pointer += 512;
  NN_initTensor(&model->decode_conv1_0_1_running_var, 1, (size_t[]){512}, DTYPE_F32, array_pointer);
  array_pointer += 512;
  NN_initTensor(&model->decode_conv1_0_1, 4, (size_t[]){1, 512, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv1_0_2
  NN_initTensor(&model->decode_conv1_0_2, 4, (size_t[]){1, 512, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv1_1_0
  NN_initTensor(&model->decode_conv1_1_0_weight, 4, (size_t[]){200, 512, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 102400;
  NN_initTensor(&model->decode_conv1_1_0, 4, (size_t[]){1, 200, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv1_1_1
  NN_initTensor(&model->decode_conv1_1_1_weight, 1, (size_t[]){200}, DTYPE_F32, array_pointer);
  array_pointer += 200;
  NN_initTensor(&model->decode_conv1_1_1_bias, 1, (size_t[]){200}, DTYPE_F32, array_pointer);
  array_pointer += 200;
  NN_initTensor(&model->decode_conv1_1_1_running_mean, 1, (size_t[]){200}, DTYPE_F32, array_pointer);
  array_pointer += 200;
  NN_initTensor(&model->decode_conv1_1_1_running_var, 1, (size_t[]){200}, DTYPE_F32, array_pointer);
  array_pointer += 200;
  NN_initTensor(&model->decode_conv1_1_1, 4, (size_t[]){1, 200, 7, 7}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv1_1_2
  NN_initTensor(&model->decode_conv1_1_2, 4, (size_t[]){1, 200, 7, 7}, DTYPE_F32, NULL);
  NN_initTensor(&model->interpolate, 4, (size_t[]){1, 200, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv2_0_0
  NN_initTensor(&model->decode_conv2_0_0_weight, 4, (size_t[]){200, 1, 5, 5}, DTYPE_F32, array_pointer);
  array_pointer += 5000;
  NN_initTensor(&model->decode_conv2_0_0, 4, (size_t[]){1, 200, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv2_0_1
  NN_initTensor(&model->decode_conv2_0_1_weight, 1, (size_t[]){200}, DTYPE_F32, array_pointer);
  array_pointer += 200;
  NN_initTensor(&model->decode_conv2_0_1_bias, 1, (size_t[]){200}, DTYPE_F32, array_pointer);
  array_pointer += 200;
  NN_initTensor(&model->decode_conv2_0_1_running_mean, 1, (size_t[]){200}, DTYPE_F32, array_pointer);
  array_pointer += 200;
  NN_initTensor(&model->decode_conv2_0_1_running_var, 1, (size_t[]){200}, DTYPE_F32, array_pointer);
  array_pointer += 200;
  NN_initTensor(&model->decode_conv2_0_1, 4, (size_t[]){1, 200, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv2_0_2
  NN_initTensor(&model->decode_conv2_0_2, 4, (size_t[]){1, 200, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv2_1_0
  NN_initTensor(&model->decode_conv2_1_0_weight, 4, (size_t[]){256, 200, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 51200;
  NN_initTensor(&model->decode_conv2_1_0, 4, (size_t[]){1, 256, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv2_1_1
  NN_initTensor(&model->decode_conv2_1_1_weight, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->decode_conv2_1_1_bias, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->decode_conv2_1_1_running_mean, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->decode_conv2_1_1_running_var, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->decode_conv2_1_1, 4, (size_t[]){1, 256, 14, 14}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv2_1_2
  NN_initTensor(&model->decode_conv2_1_2, 4, (size_t[]){1, 256, 14, 14}, DTYPE_F32, NULL);
  NN_initTensor(&model->interpolate_1, 4, (size_t[]){1, 256, 28, 28}, DTYPE_F32, NULL);
  NN_initTensor(&model->add, 4, (size_t[]){1, 256, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv3_0_0
  NN_initTensor(&model->decode_conv3_0_0_weight, 4, (size_t[]){256, 1, 5, 5}, DTYPE_F32, array_pointer);
  array_pointer += 6400;
  NN_initTensor(&model->decode_conv3_0_0, 4, (size_t[]){1, 256, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv3_0_1
  NN_initTensor(&model->decode_conv3_0_1_weight, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->decode_conv3_0_1_bias, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->decode_conv3_0_1_running_mean, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->decode_conv3_0_1_running_var, 1, (size_t[]){256}, DTYPE_F32, array_pointer);
  array_pointer += 256;
  NN_initTensor(&model->decode_conv3_0_1, 4, (size_t[]){1, 256, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv3_0_2
  NN_initTensor(&model->decode_conv3_0_2, 4, (size_t[]){1, 256, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv3_1_0
  NN_initTensor(&model->decode_conv3_1_0_weight, 4, (size_t[]){120, 256, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 30720;
  NN_initTensor(&model->decode_conv3_1_0, 4, (size_t[]){1, 120, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv3_1_1
  NN_initTensor(&model->decode_conv3_1_1_weight, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->decode_conv3_1_1_bias, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->decode_conv3_1_1_running_mean, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->decode_conv3_1_1_running_var, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->decode_conv3_1_1, 4, (size_t[]){1, 120, 28, 28}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv3_1_2
  NN_initTensor(&model->decode_conv3_1_2, 4, (size_t[]){1, 120, 28, 28}, DTYPE_F32, NULL);
  NN_initTensor(&model->interpolate_2, 4, (size_t[]){1, 120, 56, 56}, DTYPE_F32, NULL);
  NN_initTensor(&model->add_1, 4, (size_t[]){1, 120, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv4_0_0
  NN_initTensor(&model->decode_conv4_0_0_weight, 4, (size_t[]){120, 1, 5, 5}, DTYPE_F32, array_pointer);
  array_pointer += 3000;
  NN_initTensor(&model->decode_conv4_0_0, 4, (size_t[]){1, 120, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv4_0_1
  NN_initTensor(&model->decode_conv4_0_1_weight, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->decode_conv4_0_1_bias, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->decode_conv4_0_1_running_mean, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->decode_conv4_0_1_running_var, 1, (size_t[]){120}, DTYPE_F32, array_pointer);
  array_pointer += 120;
  NN_initTensor(&model->decode_conv4_0_1, 4, (size_t[]){1, 120, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv4_0_2
  NN_initTensor(&model->decode_conv4_0_2, 4, (size_t[]){1, 120, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv4_1_0
  NN_initTensor(&model->decode_conv4_1_0_weight, 4, (size_t[]){56, 120, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 6720;
  NN_initTensor(&model->decode_conv4_1_0, 4, (size_t[]){1, 56, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv4_1_1
  NN_initTensor(&model->decode_conv4_1_1_weight, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->decode_conv4_1_1_bias, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->decode_conv4_1_1_running_mean, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->decode_conv4_1_1_running_var, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->decode_conv4_1_1, 4, (size_t[]){1, 56, 56, 56}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv4_1_2
  NN_initTensor(&model->decode_conv4_1_2, 4, (size_t[]){1, 56, 56, 56}, DTYPE_F32, NULL);
  NN_initTensor(&model->interpolate_3, 4, (size_t[]){1, 56, 112, 112}, DTYPE_F32, NULL);
  NN_initTensor(&model->add_2, 4, (size_t[]){1, 56, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv5_0_0
  NN_initTensor(&model->decode_conv5_0_0_weight, 4, (size_t[]){56, 1, 5, 5}, DTYPE_F32, array_pointer);
  array_pointer += 1400;
  NN_initTensor(&model->decode_conv5_0_0, 4, (size_t[]){1, 56, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv5_0_1
  NN_initTensor(&model->decode_conv5_0_1_weight, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->decode_conv5_0_1_bias, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->decode_conv5_0_1_running_mean, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->decode_conv5_0_1_running_var, 1, (size_t[]){56}, DTYPE_F32, array_pointer);
  array_pointer += 56;
  NN_initTensor(&model->decode_conv5_0_1, 4, (size_t[]){1, 56, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv5_0_2
  NN_initTensor(&model->decode_conv5_0_2, 4, (size_t[]){1, 56, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv5_1_0
  NN_initTensor(&model->decode_conv5_1_0_weight, 4, (size_t[]){16, 56, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 896;
  NN_initTensor(&model->decode_conv5_1_0, 4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv5_1_1
  NN_initTensor(&model->decode_conv5_1_1_weight, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->decode_conv5_1_1_bias, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->decode_conv5_1_1_running_mean, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->decode_conv5_1_1_running_var, 1, (size_t[]){16}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->decode_conv5_1_1, 4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv5_1_2
  NN_initTensor(&model->decode_conv5_1_2, 4, (size_t[]){1, 16, 112, 112}, DTYPE_F32, NULL);
  NN_initTensor(&model->interpolate_4, 4, (size_t[]){1, 16, 224, 224}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: decode_conv6_0
  NN_initTensor(&model->decode_conv6_0_weight, 4, (size_t[]){1, 16, 1, 1}, DTYPE_F32, array_pointer);
  array_pointer += 16;
  NN_initTensor(&model->decode_conv6_0, 4, (size_t[]){1, 1, 224, 224}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: decode_conv6_1
  NN_initTensor(&model->decode_conv6_1_weight, 1, (size_t[]){1}, DTYPE_F32, array_pointer);
  array_pointer += 1;
  NN_initTensor(&model->decode_conv6_1_bias, 1, (size_t[]){1}, DTYPE_F32, array_pointer);
  array_pointer += 1;
  NN_initTensor(&model->decode_conv6_1_running_mean, 1, (size_t[]){1}, DTYPE_F32, array_pointer);
  array_pointer += 1;
  NN_initTensor(&model->decode_conv6_1_running_var, 1, (size_t[]){1}, DTYPE_F32, array_pointer);
  array_pointer += 1;
  NN_initTensor(&model->decode_conv6_1, 4, (size_t[]){1, 1, 224, 224}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU'>: decode_conv6_2
  NN_initTensor(&model->decode_conv6_2, 4, (size_t[]){1, 1, 224, 224}, DTYPE_F32, NULL);

}


/**
 * Forward pass of the model
 */
void forward(Model *model) {
  NN_Conv2d_F32(
    &model->conv0_0, &model->x,
    &model->conv0_0_weight, NULL, (size_t[]){2, 2}, (size_t[]){1, 1}, 1);
  NN_BatchNorm2d_F32(
    &model->conv0_1, &model->conv0_0,
    &model->conv0_1_weight, &model->conv0_1_bias, 
    1e-05, &model->conv0_1_running_mean, &model->conv0_1_running_var);
  NN_ReLU6_F32(&model->conv0_2, &model->conv0_1);
  NN_Conv2d_F32(
    &model->conv1_0, &model->conv0_2,
    &model->conv1_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){1, 1}, 16);
  NN_BatchNorm2d_F32(
    &model->conv1_1, &model->conv1_0,
    &model->conv1_1_weight, &model->conv1_1_bias, 
    1e-05, &model->conv1_1_running_mean, &model->conv1_1_running_var);
  NN_ReLU6_F32(&model->conv1_2, &model->conv1_1);
  NN_Conv2d_F32(
    &model->conv1_3, &model->conv1_2,
    &model->conv1_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv1_4, &model->conv1_3,
    &model->conv1_4_weight, &model->conv1_4_bias, 
    1e-05, &model->conv1_4_running_mean, &model->conv1_4_running_var);
  NN_ReLU6_F32(&model->conv1_5, &model->conv1_4);
  NN_Conv2d_F32(
    &model->conv2_0, &model->conv1_5,
    &model->conv2_0_weight, NULL, (size_t[]){2, 2}, (size_t[]){1, 1}, 56);
  NN_BatchNorm2d_F32(
    &model->conv2_1, &model->conv2_0,
    &model->conv2_1_weight, &model->conv2_1_bias, 
    1e-05, &model->conv2_1_running_mean, &model->conv2_1_running_var);
  NN_ReLU6_F32(&model->conv2_2, &model->conv2_1);
  NN_Conv2d_F32(
    &model->conv2_3, &model->conv2_2,
    &model->conv2_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv2_4, &model->conv2_3,
    &model->conv2_4_weight, &model->conv2_4_bias, 
    1e-05, &model->conv2_4_running_mean, &model->conv2_4_running_var);
  NN_ReLU6_F32(&model->conv2_5, &model->conv2_4);
  NN_Conv2d_F32(
    &model->conv3_0, &model->conv2_5,
    &model->conv3_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){1, 1}, 88);
  NN_BatchNorm2d_F32(
    &model->conv3_1, &model->conv3_0,
    &model->conv3_1_weight, &model->conv3_1_bias, 
    1e-05, &model->conv3_1_running_mean, &model->conv3_1_running_var);
  NN_ReLU6_F32(&model->conv3_2, &model->conv3_1);
  NN_Conv2d_F32(
    &model->conv3_3, &model->conv3_2,
    &model->conv3_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv3_4, &model->conv3_3,
    &model->conv3_4_weight, &model->conv3_4_bias, 
    1e-05, &model->conv3_4_running_mean, &model->conv3_4_running_var);
  NN_ReLU6_F32(&model->conv3_5, &model->conv3_4);
  NN_Conv2d_F32(
    &model->conv4_0, &model->conv3_5,
    &model->conv4_0_weight, NULL, (size_t[]){2, 2}, (size_t[]){1, 1}, 120);
  NN_BatchNorm2d_F32(
    &model->conv4_1, &model->conv4_0,
    &model->conv4_1_weight, &model->conv4_1_bias, 
    1e-05, &model->conv4_1_running_mean, &model->conv4_1_running_var);
  NN_ReLU6_F32(&model->conv4_2, &model->conv4_1);
  NN_Conv2d_F32(
    &model->conv4_3, &model->conv4_2,
    &model->conv4_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv4_4, &model->conv4_3,
    &model->conv4_4_weight, &model->conv4_4_bias, 
    1e-05, &model->conv4_4_running_mean, &model->conv4_4_running_var);
  NN_ReLU6_F32(&model->conv4_5, &model->conv4_4);
  NN_Conv2d_F32(
    &model->conv5_0, &model->conv4_5,
    &model->conv5_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){1, 1}, 144);
  NN_BatchNorm2d_F32(
    &model->conv5_1, &model->conv5_0,
    &model->conv5_1_weight, &model->conv5_1_bias, 
    1e-05, &model->conv5_1_running_mean, &model->conv5_1_running_var);
  NN_ReLU6_F32(&model->conv5_2, &model->conv5_1);
  NN_Conv2d_F32(
    &model->conv5_3, &model->conv5_2,
    &model->conv5_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv5_4, &model->conv5_3,
    &model->conv5_4_weight, &model->conv5_4_bias, 
    1e-05, &model->conv5_4_running_mean, &model->conv5_4_running_var);
  NN_ReLU6_F32(&model->conv5_5, &model->conv5_4);
  NN_Conv2d_F32(
    &model->conv6_0, &model->conv5_5,
    &model->conv6_0_weight, NULL, (size_t[]){2, 2}, (size_t[]){1, 1}, 256);
  NN_BatchNorm2d_F32(
    &model->conv6_1, &model->conv6_0,
    &model->conv6_1_weight, &model->conv6_1_bias, 
    1e-05, &model->conv6_1_running_mean, &model->conv6_1_running_var);
  NN_ReLU6_F32(&model->conv6_2, &model->conv6_1);
  NN_Conv2d_F32(
    &model->conv6_3, &model->conv6_2,
    &model->conv6_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv6_4, &model->conv6_3,
    &model->conv6_4_weight, &model->conv6_4_bias, 
    1e-05, &model->conv6_4_running_mean, &model->conv6_4_running_var);
  NN_ReLU6_F32(&model->conv6_5, &model->conv6_4);
  NN_Conv2d_F32(
    &model->conv7_0, &model->conv6_5,
    &model->conv7_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){1, 1}, 408);
  NN_BatchNorm2d_F32(
    &model->conv7_1, &model->conv7_0,
    &model->conv7_1_weight, &model->conv7_1_bias, 
    1e-05, &model->conv7_1_running_mean, &model->conv7_1_running_var);
  NN_ReLU6_F32(&model->conv7_2, &model->conv7_1);
  NN_Conv2d_F32(
    &model->conv7_3, &model->conv7_2,
    &model->conv7_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv7_4, &model->conv7_3,
    &model->conv7_4_weight, &model->conv7_4_bias, 
    1e-05, &model->conv7_4_running_mean, &model->conv7_4_running_var);
  NN_ReLU6_F32(&model->conv7_5, &model->conv7_4);
  NN_Conv2d_F32(
    &model->conv8_0, &model->conv7_5,
    &model->conv8_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){1, 1}, 376);
  NN_BatchNorm2d_F32(
    &model->conv8_1, &model->conv8_0,
    &model->conv8_1_weight, &model->conv8_1_bias, 
    1e-05, &model->conv8_1_running_mean, &model->conv8_1_running_var);
  NN_ReLU6_F32(&model->conv8_2, &model->conv8_1);
  NN_Conv2d_F32(
    &model->conv8_3, &model->conv8_2,
    &model->conv8_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv8_4, &model->conv8_3,
    &model->conv8_4_weight, &model->conv8_4_bias, 
    1e-05, &model->conv8_4_running_mean, &model->conv8_4_running_var);
  NN_ReLU6_F32(&model->conv8_5, &model->conv8_4);
  NN_Conv2d_F32(
    &model->conv9_0, &model->conv8_5,
    &model->conv9_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){1, 1}, 272);
  NN_BatchNorm2d_F32(
    &model->conv9_1, &model->conv9_0,
    &model->conv9_1_weight, &model->conv9_1_bias, 
    1e-05, &model->conv9_1_running_mean, &model->conv9_1_running_var);
  NN_ReLU6_F32(&model->conv9_2, &model->conv9_1);
  NN_Conv2d_F32(
    &model->conv9_3, &model->conv9_2,
    &model->conv9_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv9_4, &model->conv9_3,
    &model->conv9_4_weight, &model->conv9_4_bias, 
    1e-05, &model->conv9_4_running_mean, &model->conv9_4_running_var);
  NN_ReLU6_F32(&model->conv9_5, &model->conv9_4);
  NN_Conv2d_F32(
    &model->conv10_0, &model->conv9_5,
    &model->conv10_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){1, 1}, 288);
  NN_BatchNorm2d_F32(
    &model->conv10_1, &model->conv10_0,
    &model->conv10_1_weight, &model->conv10_1_bias, 
    1e-05, &model->conv10_1_running_mean, &model->conv10_1_running_var);
  NN_ReLU6_F32(&model->conv10_2, &model->conv10_1);
  NN_Conv2d_F32(
    &model->conv10_3, &model->conv10_2,
    &model->conv10_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv10_4, &model->conv10_3,
    &model->conv10_4_weight, &model->conv10_4_bias, 
    1e-05, &model->conv10_4_running_mean, &model->conv10_4_running_var);
  NN_ReLU6_F32(&model->conv10_5, &model->conv10_4);
  NN_Conv2d_F32(
    &model->conv11_0, &model->conv10_5,
    &model->conv11_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){1, 1}, 296);
  NN_BatchNorm2d_F32(
    &model->conv11_1, &model->conv11_0,
    &model->conv11_1_weight, &model->conv11_1_bias, 
    1e-05, &model->conv11_1_running_mean, &model->conv11_1_running_var);
  NN_ReLU6_F32(&model->conv11_2, &model->conv11_1);
  NN_Conv2d_F32(
    &model->conv11_3, &model->conv11_2,
    &model->conv11_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv11_4, &model->conv11_3,
    &model->conv11_4_weight, &model->conv11_4_bias, 
    1e-05, &model->conv11_4_running_mean, &model->conv11_4_running_var);
  NN_ReLU6_F32(&model->conv11_5, &model->conv11_4);
  NN_Conv2d_F32(
    &model->conv12_0, &model->conv11_5,
    &model->conv12_0_weight, NULL, (size_t[]){2, 2}, (size_t[]){1, 1}, 328);
  NN_BatchNorm2d_F32(
    &model->conv12_1, &model->conv12_0,
    &model->conv12_1_weight, &model->conv12_1_bias, 
    1e-05, &model->conv12_1_running_mean, &model->conv12_1_running_var);
  NN_ReLU6_F32(&model->conv12_2, &model->conv12_1);
  NN_Conv2d_F32(
    &model->conv12_3, &model->conv12_2,
    &model->conv12_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv12_4, &model->conv12_3,
    &model->conv12_4_weight, &model->conv12_4_bias, 
    1e-05, &model->conv12_4_running_mean, &model->conv12_4_running_var);
  NN_ReLU6_F32(&model->conv12_5, &model->conv12_4);
  NN_Conv2d_F32(
    &model->conv13_0, &model->conv12_5,
    &model->conv13_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){1, 1}, 480);
  NN_BatchNorm2d_F32(
    &model->conv13_1, &model->conv13_0,
    &model->conv13_1_weight, &model->conv13_1_bias, 
    1e-05, &model->conv13_1_running_mean, &model->conv13_1_running_var);
  NN_ReLU6_F32(&model->conv13_2, &model->conv13_1);
  NN_Conv2d_F32(
    &model->conv13_3, &model->conv13_2,
    &model->conv13_3_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->conv13_4, &model->conv13_3,
    &model->conv13_4_weight, &model->conv13_4_bias, 
    1e-05, &model->conv13_4_running_mean, &model->conv13_4_running_var);
  NN_ReLU6_F32(&model->conv13_5, &model->conv13_4);
  NN_Conv2d_F32(
    &model->decode_conv1_0_0, &model->conv13_5,
    &model->decode_conv1_0_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){2, 2}, 512);
  NN_BatchNorm2d_F32(
    &model->decode_conv1_0_1, &model->decode_conv1_0_0,
    &model->decode_conv1_0_1_weight, &model->decode_conv1_0_1_bias, 
    1e-05, &model->decode_conv1_0_1_running_mean, &model->decode_conv1_0_1_running_var);
  NN_ReLU_F32(&model->decode_conv1_0_2, &model->decode_conv1_0_1);
  NN_Conv2d_F32(
    &model->decode_conv1_1_0, &model->decode_conv1_0_2,
    &model->decode_conv1_1_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->decode_conv1_1_1, &model->decode_conv1_1_0,
    &model->decode_conv1_1_1_weight, &model->decode_conv1_1_1_bias, 
    1e-05, &model->decode_conv1_1_1_running_mean, &model->decode_conv1_1_1_running_var);
  NN_ReLU_F32(&model->decode_conv1_1_2, &model->decode_conv1_1_1);
  // F.interpolate
  NN_interpolate_F32(&model->interpolate, &model->decode_conv1_1_2, (float []){2, 2});

  NN_Conv2d_F32(
    &model->decode_conv2_0_0, &model->interpolate,
    &model->decode_conv2_0_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){2, 2}, 200);
  NN_BatchNorm2d_F32(
    &model->decode_conv2_0_1, &model->decode_conv2_0_0,
    &model->decode_conv2_0_1_weight, &model->decode_conv2_0_1_bias, 
    1e-05, &model->decode_conv2_0_1_running_mean, &model->decode_conv2_0_1_running_var);
  NN_ReLU_F32(&model->decode_conv2_0_2, &model->decode_conv2_0_1);
  NN_Conv2d_F32(
    &model->decode_conv2_1_0, &model->decode_conv2_0_2,
    &model->decode_conv2_1_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->decode_conv2_1_1, &model->decode_conv2_1_0,
    &model->decode_conv2_1_1_weight, &model->decode_conv2_1_1_bias, 
    1e-05, &model->decode_conv2_1_1_running_mean, &model->decode_conv2_1_1_running_var);
  NN_ReLU_F32(&model->decode_conv2_1_2, &model->decode_conv2_1_1);
  // F.interpolate_1
  NN_interpolate_F32(&model->interpolate_1, &model->decode_conv2_1_2, (float []){2, 2});

  // F.add
  NN_add_F32(&model->add, &model->interpolate_1, &model->conv5_5);

  NN_Conv2d_F32(
    &model->decode_conv3_0_0, &model->add,
    &model->decode_conv3_0_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){2, 2}, 256);
  NN_BatchNorm2d_F32(
    &model->decode_conv3_0_1, &model->decode_conv3_0_0,
    &model->decode_conv3_0_1_weight, &model->decode_conv3_0_1_bias, 
    1e-05, &model->decode_conv3_0_1_running_mean, &model->decode_conv3_0_1_running_var);
  NN_ReLU_F32(&model->decode_conv3_0_2, &model->decode_conv3_0_1);
  NN_Conv2d_F32(
    &model->decode_conv3_1_0, &model->decode_conv3_0_2,
    &model->decode_conv3_1_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->decode_conv3_1_1, &model->decode_conv3_1_0,
    &model->decode_conv3_1_1_weight, &model->decode_conv3_1_1_bias, 
    1e-05, &model->decode_conv3_1_1_running_mean, &model->decode_conv3_1_1_running_var);
  NN_ReLU_F32(&model->decode_conv3_1_2, &model->decode_conv3_1_1);
  // F.interpolate_2
  NN_interpolate_F32(&model->interpolate_2, &model->decode_conv3_1_2, (float []){2, 2});

  // F.add_1
  NN_add_F32(&model->add_1, &model->interpolate_2, &model->conv3_5);

  NN_Conv2d_F32(
    &model->decode_conv4_0_0, &model->add_1,
    &model->decode_conv4_0_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){2, 2}, 120);
  NN_BatchNorm2d_F32(
    &model->decode_conv4_0_1, &model->decode_conv4_0_0,
    &model->decode_conv4_0_1_weight, &model->decode_conv4_0_1_bias, 
    1e-05, &model->decode_conv4_0_1_running_mean, &model->decode_conv4_0_1_running_var);
  NN_ReLU_F32(&model->decode_conv4_0_2, &model->decode_conv4_0_1);
  NN_Conv2d_F32(
    &model->decode_conv4_1_0, &model->decode_conv4_0_2,
    &model->decode_conv4_1_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->decode_conv4_1_1, &model->decode_conv4_1_0,
    &model->decode_conv4_1_1_weight, &model->decode_conv4_1_1_bias, 
    1e-05, &model->decode_conv4_1_1_running_mean, &model->decode_conv4_1_1_running_var);
  NN_ReLU_F32(&model->decode_conv4_1_2, &model->decode_conv4_1_1);
  // F.interpolate_3
  NN_interpolate_F32(&model->interpolate_3, &model->decode_conv4_1_2, (float []){2, 2});

  // F.add_2
  NN_add_F32(&model->add_2, &model->interpolate_3, &model->conv1_5);

  NN_Conv2d_F32(
    &model->decode_conv5_0_0, &model->add_2,
    &model->decode_conv5_0_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){2, 2}, 56);
  NN_BatchNorm2d_F32(
    &model->decode_conv5_0_1, &model->decode_conv5_0_0,
    &model->decode_conv5_0_1_weight, &model->decode_conv5_0_1_bias, 
    1e-05, &model->decode_conv5_0_1_running_mean, &model->decode_conv5_0_1_running_var);
  NN_ReLU_F32(&model->decode_conv5_0_2, &model->decode_conv5_0_1);
  NN_Conv2d_F32(
    &model->decode_conv5_1_0, &model->decode_conv5_0_2,
    &model->decode_conv5_1_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->decode_conv5_1_1, &model->decode_conv5_1_0,
    &model->decode_conv5_1_1_weight, &model->decode_conv5_1_1_bias, 
    1e-05, &model->decode_conv5_1_1_running_mean, &model->decode_conv5_1_1_running_var);
  NN_ReLU_F32(&model->decode_conv5_1_2, &model->decode_conv5_1_1);
  // F.interpolate_4
  NN_interpolate_F32(&model->interpolate_4, &model->decode_conv5_1_2, (float []){2, 2});

  NN_Conv2d_F32(
    &model->decode_conv6_0, &model->interpolate_4,
    &model->decode_conv6_0_weight, NULL, (size_t[]){1, 1}, (size_t[]){0, 0}, 1);
  NN_BatchNorm2d_F32(
    &model->decode_conv6_1, &model->decode_conv6_0,
    &model->decode_conv6_1_weight, &model->decode_conv6_1_bias, 
    1e-05, &model->decode_conv6_1_running_mean, &model->decode_conv6_1_running_var);
  NN_ReLU_F32(&model->decode_conv6_2, &model->decode_conv6_1);

}

#endif