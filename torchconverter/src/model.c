
#include "nn.h"



// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "../model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {
  Tensor input;
  Tensor conv0_0_0_weight;
  Tensor conv0_0_0_out;
  Tensor conv0_0_1_weight;
  Tensor conv0_0_1_bias;
  Tensor conv0_0_1_running_mean;
  Tensor conv0_0_1_running_var;
  Tensor conv0_0_1_out;

} Model;

/**
 * Initialize the required tensors for the model
 */
void init(Model *model) {
  float *array_pointer = (float *)model_weight_data;

  NN_initTensor(&model->input, 2, (size_t[])(1, 784), DTYPE_F32, NULL);

  // <class 'torch.nn.modules.conv.Conv2d'>: conv0_0_0
  NN_initTensor(&model->conv0_0_0_weight, 4, (size_t[]){32, 3, 3, 3}, DTYPE_F32, array_pointer);
  array_pointer += 864;
  NN_initTensor(&model->conv0_0_0_out, 4, (size_t[]){1, 32, 3, 3}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.batchnorm.BatchNorm2d'>: conv0_0_1
  NN_initTensor(&model->conv0_0_1_weight, 1, (size_t[]){32}, DTYPE_F32, array_pointer);
  array_pointer += 32;
  NN_initTensor(&model->conv0_0_1_bias, 1, (size_t[]){32}, DTYPE_F32, array_pointer);
  array_pointer += 32;
  NN_initTensor(&model->conv0_0_1_running_mean, 1, (size_t[]){32}, DTYPE_F32, array_pointer);
  array_pointer += 32;
  NN_initTensor(&model->conv0_0_1_running_var, 1, (size_t[]){32}, DTYPE_F32, array_pointer);
  array_pointer += 32;
  NN_initTensor(&model->conv0_0_1_out, 4, (size_t[]){1, 32, 16, 16}, DTYPE_F32, NULL);

  // <class 'torch.nn.modules.activation.ReLU6'>: conv0_0_2

}


/**
 * Forward pass of the model
 */
void forward(Model *model) {
  NN_Conv2D_F32(
    &model->conv0_0_0_out, &model->conv0_0_0_out,
    &model->conv0_0_0_weight, NULL, (size_t[]){3, 3}, (size_t[]){2, 2}, 1);
  NN_BatchNorm2D_F32(&model->conv0_0_1_out, &model->conv0_0_1_out, &model->conv0_0_1_weight, &model->conv0_0_1_bias, &model->conv0_0_1_running_mean, &model->conv0_0_1_running_var, 1e-05);
  NN_ReLU6Inplace_F32(&model->conv0_0_1_out);

}