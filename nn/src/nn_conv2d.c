
#include "nn_conv2d.h"

#ifdef GEMMINI
  #include "gemmini/gemmini.h"
#endif


void NN_NCHWToNHWC(Tensor *out, Tensor *in) {
  assert(in->ndim == 4);
  assert(out->ndim == 4);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->shape[0] == in->shape[0]); // batch size must match
  assert(out->shape[1] == in->shape[2]); // height must match
  assert(out->shape[2] == in->shape[3]); // width must match
  assert(out->shape[3] == in->shape[1]); // channels must match

  size_t batch_size = in->shape[0];
  size_t height = in->shape[2];
  size_t width = in->shape[3];
  size_t channels = in->shape[1];

  for (size_t n = 0; n < batch_size; n += 1) {
    for (size_t c = 0; c < channels; c += 1) {
      for (size_t h = 0; h < height; h += 1) {
        for (size_t w = 0; w < width; w += 1) {
          size_t nchw_index = n * channels * height * width + c * height * width + h * width + w;
          size_t nhwc_index = n * height * width * channels + h * width * channels + w * channels + c;
          ((float *)out->data)[nhwc_index] = ((float *)in->data)[nchw_index];
        }
      }
    }
  }
}

void NN_NHWCToNCHW(Tensor *out, Tensor *in) {
  assert(in->ndim == 4);
  assert(out->ndim == 4);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(out->shape[0] == in->shape[0]); // batch size must match
  assert(out->shape[1] == in->shape[3]); // channels must match
  assert(out->shape[2] == in->shape[1]); // height must match
  assert(out->shape[3] == in->shape[2]); // width must match

  size_t batch_size = in->shape[0];
  size_t height = in->shape[1];
  size_t width = in->shape[2];
  size_t channels = in->shape[3];

  for (size_t n = 0; n < batch_size; n += 1) {
    for (size_t c = 0; c < channels; c += 1) {
      for (size_t h = 0; h < height; h += 1) {
        for (size_t w = 0; w < width; w += 1) {
          size_t nhwc_index = n * height * width * channels + h * width * channels + w * channels + c;
          size_t nchw_index = n * channels * height * width + c * height * width + h * width + w;
          ((float *)out->data)[nchw_index] = ((float *)in->data)[nhwc_index];
        }
      }
    }
  }
}

void NN_Conv2d(
  Tensor *out, Tensor *in,
  Tensor *weight, Tensor *bias,
  const size_t *stride, const size_t *padding, const size_t *dilation, size_t groups) {
  assert(in->ndim == 4);
  assert(out->ndim == 4);
  assert(in->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
  assert(weight->dtype == DTYPE_F32);
  if (bias != NULL) {
    assert(bias->ndim == 1);
    assert(bias->dtype == DTYPE_F32);
  }

  size_t batch_size = in->shape[0];
  size_t in_height = in->shape[1];
  size_t in_width = in->shape[2];
  size_t in_channels = in->shape[3];
  
  size_t out_height = out->shape[1];
  size_t out_width = out->shape[2];
  size_t out_channels = out->shape[3];

  size_t kernel_height = weight->shape[0];
  size_t kernel_width = weight->shape[1];
  size_t stride_height = stride[0];
  size_t stride_width = stride[1];
  size_t padding_height = padding[0];
  size_t padding_width = padding[1];
  size_t dilation_height = dilation[0];
  size_t dilation_width = dilation[1];

  assert(out->shape[0] == batch_size);
  assert(weight->shape[3] == out_channels);
  assert(weight->shape[2] * groups == in_channels);
  assert(out_height == (in_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) / stride_height + 1);
  assert(out_width == (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) / stride_width + 1);
  assert(groups > 0);
  assert(in_channels % groups == 0);
  assert(out_channels % groups == 0);


  // Initialize output tensor to zeros
  memset(out->data, 0, batch_size * out_height * out_width * out_channels * sizeof(float));

  #ifdef GEMMINI
    if (!bias) {
      bias = NN_zeros(1, (size_t []){ out_channels }, DTYPE_F32);
    }

    if (groups == 1) {
      tiled_conv_auto(
          batch_size, in_height, in_width, in_channels,
          out_channels, out_height, out_width,
          stride_height, dilation_height, 1, padding_height, kernel_height, 
          0, 0, 0, 0, 0,

          in->data,
          weight->data,
          bias->data,
          out->data,

          NO_ACTIVATION, ACC_SCALE_IDENTITY,
          0, 0, 0,

          WS);
    }
    else {
      assert(weight->shape[2] == 1);

      Tensor *in_nchw = NN_tensor(4, (size_t[]){batch_size, in_channels, in_height, in_width}, DTYPE_F32, NULL);
      Tensor *out_nchw = NN_tensor(4, (size_t[]){batch_size, out_channels, out_height, out_width}, DTYPE_F32, NULL);
      
      Tensor *weight_1hwc = NN_tensor(4, (size_t[]){1, kernel_height, kernel_width, out_channels}, DTYPE_F32, weight->data);
      Tensor *weight_1chw = NN_tensor(4, (size_t[]){1, out_channels, kernel_height, kernel_width}, DTYPE_F32, NULL);

      // printf("in:\t");
      // NN_printShape(in);
      // printf("\n");
      // NN_printf(in);
      // printf("in_nchw:\t");
      // NN_printShape(in_nchw);
      // printf("\n");
      // NN_printf(in_nchw);

      
      // printf("weight_1hwc:\t");
      // NN_printShape(weight_1hwc);
      // printf("\n");
      // NN_printf(weight_1hwc);
      // printf("weight_1chw:\t");
      // NN_printShape(weight_1chw);
      // printf("\n");
      // NN_printf(weight_1chw);

      NN_NHWCToNCHW(in_nchw, in);
      NN_NHWCToNCHW(weight_1chw, weight_1hwc);


      

      for (size_t g = 0; g < groups; g += 1) {
        tiled_conv_auto(
          batch_size, in_height, in_width, 1,
          1, out_height, out_width,
          stride_height, dilation_height, 1, padding_height, kernel_height, 
          0, 0, 0, 0, 0,

          ((float *)in_nchw->data) + g * in_height * in_width,
          ((float *)weight_1chw->data) + g * kernel_height * kernel_width,
          ((float *)bias->data) + g,
          ((float *)out_nchw->data) + g * out_height * out_width,

          NO_ACTIVATION, ACC_SCALE_IDENTITY,
          0, 0, 0,

          WS);

          // printf("channel group %d\n", g);
          // NN_printf(out_nchw);
          // printf("\n\n");
      }

      NN_NCHWToNHWC(out, out_nchw);
    }


  #else
    for (size_t n = 0; n < batch_size; n += 1) {
      for (size_t g = 0; g < groups; g += 1) {
        for (size_t oc = 0; oc < out_channels / groups; oc += 1) {
          for (size_t oh = 0; oh < out_height; oh += 1) {
            for (size_t ow = 0; ow < out_width; ow += 1) {
              float sum = 0.f;
              if (bias != NULL) {
                sum = ((float *)bias->data)[g * (out_channels / groups) + oc];
              }
              for (size_t ic = 0; ic < in_channels / groups; ic += 1) {
                for (size_t kh = 0; kh < kernel_height; kh += 1) {
                  for (size_t kw = 0; kw < kernel_width; kw += 1) {
                    int ih = oh * stride_height + kh * dilation_height - padding_height;
                    int iw = ow * stride_width + kw * dilation_width - padding_width;
                    if (ih < (int)in_height && iw < (int)in_width && ih >= 0 && iw >= 0) {
                      size_t in_idx = n * in_height * in_width * in_channels
                                  + ih * in_width * in_channels
                                  + iw * in_channels
                                  + g * (in_channels / groups)
                                  + ic;
                      size_t weight_idx = kh * kernel_width * in_channels * out_channels / groups
                                  + kw * in_channels * out_channels / groups
                                  + ic * out_channels / groups
                                  + oc;
                      sum += ((float *)in->data)[in_idx] * ((float *)weight->data)[weight_idx + g * (in_channels / groups) * (kernel_height * kernel_width * out_channels / groups)];
                    }
                  }
                }
              }
              size_t out_idx = n * out_height * out_width * out_channels
                            + oh * out_width * out_channels
                            + ow * out_channels
                            + g * (out_channels / groups) + oc;
              ((float *)out->data)[out_idx] = sum;
            }
          }
        }
      }
    }
  #endif
}
