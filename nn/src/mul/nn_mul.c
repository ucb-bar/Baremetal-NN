
#include "nn_mul.h"

void NN_mul_F32(Tensor *out, Tensor *in, float coeff) {
  assert(out->ndim == in->ndim);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  
  uint8_t *in_ptr = in->data;
  uint8_t *out_ptr = out->data;
  
  switch (in->ndim) {
    case 1:
      assert(in->shape[0] == out->shape[0]);

      for (size_t i = 0; i < in->shape[0]; i += 1) {
        *((float *)out_ptr) = *((float *)in_ptr) * coeff;
        in_ptr += in->strides[0];
        out_ptr += out->strides[0];
      }
      return;
    case 2:
      assert(in->shape[0] == out->shape[0]);
      assert(in->shape[1] == out->shape[1]);

      for (size_t i = 0; i < in->shape[0]; i += 1) {
        for (size_t j = 0; j < in->shape[1]; j += 1) {
          *((float *)out_ptr) = *((float *)in_ptr) * coeff;
          in_ptr += in->strides[1];
          out_ptr += out->strides[1];
        }
        in_ptr -= in->strides[1] * in->shape[1];
        in_ptr += in->strides[0];
        out_ptr -= out->strides[1] * out->shape[1];
        out_ptr += out->strides[0];
      }
      return;
  }
}

