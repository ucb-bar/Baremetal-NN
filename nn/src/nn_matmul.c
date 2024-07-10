
#include "nn_matmul.h"

#ifdef GEMMINI
  #include <math.h>
  #include <limits.h>
  #include <stdbool.h>

  #include "gemmini/gemmini.h"
#endif

void NN_matmul(Tensor *out, Tensor *a, Tensor *b) {

  #ifdef GEMMINI
    // // This function runs a tiled matrix multiplication, with automatically
    // // calculated tiling factors
    // static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
    //         const elem_t* A, const elem_t* B,
    //         const void * D, void * C,
    //         size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
    //         scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
    //         int act, acc_scale_t scale, acc_scale_t bert_scale,
    //         bool repeating_bias,
    //         bool transpose_A, bool transpose_B,
    //         bool full_C, bool low_D,
    //         uint8_t weightA,
    //         enum tiled_matmul_type_t tiled_matmul_type) {

    size_t dim_I = a->shape[0];
    size_t dim_J = b->shape[1];
    size_t dim_K = a->shape[1];

    size_t stride_A = dim_K;
    size_t stride_B = dim_J;
    size_t stride_D = dim_J;
    size_t stride_C = dim_J;

    scale_t A_scale_factor = 1.0;
    scale_t B_scale_factor = 1.0;
    scale_acc_t D_scale_factor = 1.0;
    
    int act = 0;
    acc_scale_t scale = 1.0;
    acc_scale_t bert_scale = 1.0;

    bool repeating_bias = false;
    bool transpose_A = false;
    bool transpose_B = false;
    bool full_C = false;
    bool low_D = false;

    tiled_matmul_auto(dim_I, dim_J, dim_K,
          a->data, b->data,
          NULL, out->data,
          stride_A, stride_B, stride_D, stride_C,
          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          repeating_bias,
          transpose_A, transpose_B,
          full_C, low_D,
          0,
          WS);
    
    return;
  #endif

  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
    // currently only support 2D matrix multiplication
    assert(a->ndim == 2);
    assert(b->ndim == 2);
    assert(a->dtype == DTYPE_F32);
    assert(b->dtype == DTYPE_F32);
    assert(out->dtype == DTYPE_F32);
    assert(a->shape[1] == b->shape[0]);
    assert(out->shape[0] == a->shape[0]);
    assert(out->shape[1] == b->shape[1]);

    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        float sum = 0;
        for (size_t k = 0; k < a->shape[1]; k += 1) {
          sum += ((float *)a->data)[i * a->shape[1] + k] * ((float *)b->data)[k * b->shape[1] + j];
        }
        ((float *)out->data)[i * out->shape[1] + j] = sum;
      }
    }
    return;
  }
  if (a->dtype == DTYPE_F16 && b->dtype == DTYPE_F16 && out->dtype == DTYPE_F16) {
    // currently only support 2D matrix multiplication
    assert(a->ndim == 2);
    assert(b->ndim == 2);
    assert(a->dtype == DTYPE_F16);
    assert(b->dtype == DTYPE_F16);
    assert(out->dtype == DTYPE_F16);
    assert(a->shape[1] == b->shape[0]);
    assert(out->shape[0] == a->shape[0]);
    assert(out->shape[1] == b->shape[1]);

    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        float sum = 0;
        for (size_t k = 0; k < a->shape[1]; k += 1) {
          sum += NN_halfToFloat(((float16_t *)a->data)[i * a->shape[1] + k]) * NN_halfToFloat(((float16_t *)b->data)[k * b->shape[1] + j]);
        }
        ((float16_t *)out->data)[i * out->shape[1] + j] = NN_floatToHalf(sum);
      }
    }
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

void NN_matmul_t(Tensor *out, Tensor *a, Tensor *b) {
  if (a->dtype == DTYPE_F16 && b->dtype == DTYPE_F16 && out->dtype == DTYPE_F16) {
    // currently only support 2D matrix multiplication
    assert(a->ndim == 2);
    assert(b->ndim == 2);
    assert(a->dtype == DTYPE_F16);
    assert(b->dtype == DTYPE_F16);
    assert(out->dtype == DTYPE_F16);
    assert(a->shape[1] == b->shape[1]);
    assert(out->shape[0] == a->shape[0]);
    assert(out->shape[1] == b->shape[0]);

    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        NN__dot_f16(a->shape[1], 
          (float16_t *)out->data + i * out->shape[1] + j, 
          (float16_t *)a->data + i * a->shape[1], 
          (float16_t *)b->data + j * b->shape[1]
          );
      }
    }
    return;
  }
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
    // currently only support 2D matrix multiplication
    assert(a->ndim == 2);
    assert(b->ndim == 2);
    assert(a->dtype == DTYPE_F32);
    assert(b->dtype == DTYPE_F32);
    assert(out->dtype == DTYPE_F32);
    assert(a->shape[1] == b->shape[1]);
    assert(out->shape[0] == a->shape[0]);
    assert(out->shape[1] == b->shape[0]);

    for (size_t i = 0; i < out->shape[0]; i += 1) {
      for (size_t j = 0; j < out->shape[1]; j += 1) {
        NN__dot_f32(a->shape[1], 
          (float *)out->data + i * out->shape[1] + j, 
          (float *)a->data + i * a->shape[1], 
          (float *)b->data + j * b->shape[1]
          );
      }
    }
    return;
  }
  printf("Unsupported operation: %s = %s @ %s\n", 
    NN_get_datatype_name(out->dtype), NN_get_datatype_name(a->dtype), NN_get_datatype_name(b->dtype)
  );
}

