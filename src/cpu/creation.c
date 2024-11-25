#include "nn.h"

__attribute__((weak)) Tensor0D_F16 *nn_tensor0d_f16(float16_t data) {
  Tensor0D_F16 *tensor = (Tensor0D_F16 *)malloc(sizeof(Tensor0D_F16));
  tensor->data = data;
}

__attribute__((weak)) Tensor0D_F32 *nn_tensor0d_f32(float data) {
  Tensor0D_F32 *tensor = (Tensor0D_F32 *)malloc(sizeof(Tensor0D_F32));
  tensor->data = data;
}

__attribute__((weak)) Tensor1D_F16 *nn_tensor1d_f16(size_t shape[1], const float16_t *data) {
  Tensor1D_F16 *tensor = (Tensor1D_F16 *)malloc(sizeof(Tensor1D_F16));
  tensor->shape[0] = shape[0];

  size_t n_bytes = shape[0] * sizeof(float16_t);
  tensor->data = (float16_t *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
}

__attribute__((weak)) Tensor1D_F32 *nn_tensor1d_f32(size_t shape[1], const float *data) {
  Tensor1D_F32 *tensor = (Tensor1D_F32 *)malloc(sizeof(Tensor1D_F32));
  tensor->shape[0] = shape[0];

  size_t n_bytes = shape[0] * sizeof(float);
  tensor->data = (float *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
}

__attribute__((weak)) Tensor2D_F16 *nn_tensor2d_f16(size_t shape[2], const float16_t *data) {
  Tensor2D_F16 *tensor = (Tensor2D_F16 *)malloc(sizeof(Tensor2D_F16));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];

  size_t n_bytes = shape[0] * shape[1] * sizeof(float16_t);
  tensor->data = (float16_t *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
}

__attribute__((weak)) Tensor2D_F32 *nn_tensor2d_f32(size_t shape[2], const float *data) {
  Tensor2D_F32 *tensor = (Tensor2D_F32 *)malloc(sizeof(Tensor2D_F32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];

  size_t n_bytes = shape[0] * shape[1] * sizeof(float);
  tensor->data = (float *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
}

__attribute__((weak)) Tensor3D_F16 *nn_tensor3d_f16(size_t shape[3], const float16_t *data) {
  Tensor3D_F16 *tensor = (Tensor3D_F16 *)malloc(sizeof(Tensor3D_F16));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];
  tensor->shape[2] = shape[2];
}

__attribute__((weak)) Tensor3D_F32 *nn_tensor3d_f32(size_t shape[3], const float *data) {
  Tensor3D_F32 *tensor = (Tensor3D_F32 *)malloc(sizeof(Tensor3D_F32));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];
  tensor->shape[2] = shape[2];
}

__attribute__((weak)) Tensor0D_F16 *nn_zeros0d_f16() {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(0);
  return tensor;
}

__attribute__((weak)) Tensor0D_F32 *nn_zeros0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(0);
  return tensor;
}

__attribute__((weak)) Tensor1D_F16 *nn_zeros1d_f16(size_t shape[1]) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

__attribute__((weak)) Tensor1D_F32 *nn_zeros1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

__attribute__((weak)) Tensor2D_F16 *nn_zeros2d_f16(size_t shape[2]) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

__attribute__((weak)) Tensor2D_F32 *nn_zeros2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

__attribute__((weak)) Tensor3D_F16 *nn_zeros3d_f16(size_t shape[3]) {
  Tensor3D_F16 *tensor = nn_tensor3d_f16(shape, NULL);
  size_t n = shape[0] * shape[1] * shape[2];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

__attribute__((weak)) Tensor3D_F32 *nn_zeros3d_f32(size_t shape[3]) {
  Tensor3D_F32 *tensor = nn_tensor3d_f32(shape, NULL);
  size_t n = shape[0] * shape[1] * shape[2];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

__attribute__((weak)) Tensor0D_F16 *nn_ones0d_f16() {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(1);
  return tensor;
}

__attribute__((weak)) Tensor0D_F32 *nn_ones0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(1);
  return tensor;
}

__attribute__((weak)) Tensor1D_F16 *nn_ones1d_f16(size_t shape[1]) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

__attribute__((weak)) Tensor1D_F32 *nn_ones1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

__attribute__((weak)) Tensor2D_F16 *nn_ones2d_f16(size_t shape[2]) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

__attribute__((weak)) Tensor2D_F32 *nn_ones2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

__attribute__((weak)) Tensor0D_F16 *nn_full0d_f16(float16_t data) {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(data);
  return tensor;
}

__attribute__((weak)) Tensor0D_F32 *nn_full0d_f32(float data) {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(data);
  return tensor;
}

__attribute__((weak)) Tensor1D_F16 *nn_full1d_f16(size_t shape[1], float16_t data) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

__attribute__((weak)) Tensor1D_F32 *nn_full1d_f32(size_t shape[1], float data) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

__attribute__((weak)) Tensor2D_F16 *nn_full2d_f16(size_t shape[2], float16_t data) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

__attribute__((weak)) Tensor2D_F32 *nn_full2d_f32(size_t shape[2], float data) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

__attribute__((weak)) Tensor0D_F16 *nn_rand0d_f16() {
  Tensor0D_F16 *tensor = nn_tensor0d_f16(as_f16(rand()));
  return tensor;
}

__attribute__((weak)) Tensor0D_F32 *nn_rand0d_f32() {
  Tensor0D_F32 *tensor = nn_tensor0d_f32(rand());
  return tensor;
}

__attribute__((weak)) Tensor1D_F16 *nn_rand1d_f16(size_t shape[1]) {
  Tensor1D_F16 *tensor = nn_tensor1d_f16(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = as_f16(rand());
  }
  return tensor;
}

__attribute__((weak)) Tensor1D_F32 *nn_rand1d_f32(size_t shape[1]) {
  Tensor1D_F32 *tensor = nn_tensor1d_f32(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
  }
  return tensor;
}

__attribute__((weak)) Tensor2D_F16 *nn_rand2d_f16(size_t shape[2]) {
  Tensor2D_F16 *tensor = nn_tensor2d_f16(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = as_f16(rand());
  }
  return tensor;
}

__attribute__((weak)) Tensor2D_F32 *nn_rand2d_f32(size_t shape[2]) {
  Tensor2D_F32 *tensor = nn_tensor2d_f32(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
  }
  return tensor;
}

