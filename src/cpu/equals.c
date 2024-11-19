#include "nn.h"


__attribute__((weak)) uint8_t nn_equals0d_f16(const Tensor0D_F16 *a, const Tensor0D_F16 *b, float rel_err) {
  return float_equal(as_f32(a->data), as_f32(b->data), rel_err);
}

__attribute__((weak)) uint8_t nn_equals0d_f32(const Tensor0D_F32 *a, const Tensor0D_F32 *b, float rel_err) {
  return float_equal(a->data, b->data, rel_err);
}

__attribute__((weak)) uint8_t nn_equals1d_f16(const Tensor1D_F16 *a, const Tensor1D_F16 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    if (!float_equal(as_f32(a->data[i]), as_f32(b->data[i]), rel_err)) {
      return 0;
    }
  }
  return 1;
}

__attribute__((weak)) uint8_t nn_equals1d_f32(const Tensor1D_F32 *a, const Tensor1D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    if (!float_equal(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}

__attribute__((weak)) uint8_t nn_equals2d_f16(const Tensor2D_F16 *a, const Tensor2D_F16 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (!float_equal(as_f32(a->data[i]), as_f32(b->data[i]), rel_err)) {
      return 0;
    }
  }
  return 1;
}

__attribute__((weak)) uint8_t nn_equals2d_f32(const Tensor2D_F32 *a, const Tensor2D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (!float_equal(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}

__attribute__((weak)) uint8_t nn_equals3d_f16(const Tensor3D_F16 *a, const Tensor3D_F16 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && a->shape[2] == b->shape[2], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1] * a->shape[2];
  for (size_t i = 0; i < n; i += 1) {
    if (!float_equal(as_f32(a->data[i]), as_f32(b->data[i]), rel_err)) {
      return 0;
    }
  }
  return 1;
}

__attribute__((weak)) uint8_t nn_equals3d_f32(const Tensor3D_F32 *a, const Tensor3D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && a->shape[2] == b->shape[2], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1] * a->shape[2];
  for (size_t i = 0; i < n; i += 1) {
    if (!float_equal(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}

__attribute__((weak)) uint8_t nn_equals4d_f16(const Tensor4D_F16 *a, const Tensor4D_F16 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && a->shape[2] == b->shape[2] && a->shape[3] == b->shape[3], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1] * a->shape[2] * a->shape[3];
  for (size_t i = 0; i < n; i += 1) {
    if (!float_equal(as_f32(a->data[i]), as_f32(b->data[i]), rel_err)) {
      return 0;
    }
  }
  return 1;
}

__attribute__((weak)) uint8_t nn_equals4d_f32(const Tensor4D_F32 *a, const Tensor4D_F32 *b, float rel_err) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1] && a->shape[2] == b->shape[2] && a->shape[3] == b->shape[3], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1] * a->shape[2] * a->shape[3];
  for (size_t i = 0; i < n; i += 1) {
    if (!float_equal(a->data[i], b->data[i], rel_err)) {
      return 0;
    }
  }
  return 1;
}
