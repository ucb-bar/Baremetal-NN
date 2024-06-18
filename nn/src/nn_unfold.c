
// #include "nn_unfold.h"

// // // src0: kernel [OC, IC, KH, KW]
// // // src1: image [N, IC, IH, IW]
// // // dst:  result [N, OH, OW, IC*KH*KW]
// // void ggml_compute_forward_im2col_f32(Tensor *out, Tensor *kernel, Tensor *image) {
// //   const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
// //   const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
// //   const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
// //   const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
// //   const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
// //   const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
// //   const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

// //   const int ith = params->ith;
// //   const int nth = params->nth;

// //   const int64_t N  = is_2D ? ne13 : ne12;
// //   const int64_t IC = is_2D ? ne12 : ne11;
// //   const int64_t IH = is_2D ? ne11 : 1;
// //   const int64_t IW = ne10;

// //   const int64_t KH = is_2D ? ne01 : 1;
// //   const int64_t KW = ne00;

// //   const int64_t OH = is_2D ? ne2 : 1;
// //   const int64_t OW = ne1;

// //   int ofs0 = is_2D ? nb13 : nb12;
// //   int ofs1 = is_2D ? nb12 : nb11;

// //   GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
// //   GGML_ASSERT(nb10 == sizeof(float));

// //   if (params->type == GGML_TASK_TYPE_INIT) {
// //       return;
// //   }

// //   if (params->type == GGML_TASK_TYPE_FINALIZE) {
// //       return;
// //   }

// //   // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// //   {
// //       float * const wdata = (float *) dst->data;

// //       for (int64_t in = 0; in < N; in++) {
// //           for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
// //               for (int64_t iow = 0; iow < OW; iow++) {
// //                   for (int64_t iic = ith; iic < IC; iic += nth) {

// //                       // micro kernel
// //                       float * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
// //                       const float * const src_data = (float *)((char *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

// //                       for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
// //                           for (int64_t ikw = 0; ikw < KW; ikw++) {
// //                               const int64_t iiw = iow*s0 + ikw*d0 - p0;
// //                               const int64_t iih = ioh*s1 + ikh*d1 - p1;

// //                               if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
// //                                   dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
// //                               } else {
// //                                   dst_data[iic*(KH*KW) + ikh*KW + ikw] = (src_data[iih*IW + iiw]);
// //                               }
// //                           }
// //                       }
// //                   }
// //               }
// //           }
// //       }
// //   }
// // }
// void NN_unfold(Tensor *data_col, Tensor *data_im, 
//             const size_t *kernel_size,
//             const size_t *stride, const size_t *padding, const size_t *dilation) {


//   size_t batch_size = data_im->shape[0];
//   size_t im_height = data_im->shape[1];
//   size_t im_width = data_im->shape[2];
//   size_t im_channels = data_im->shape[3];
//   size_t out_height = data_col->shape[1];
//   size_t out_width = data_col->shape[2];
//   size_t kernel_height = kernel_size[0];
//   size_t kernel_width = kernel_size[1];
//   size_t stride_height = stride[0];
//   size_t stride_width = stride[1];
//   size_t padding_height = padding[0];
//   size_t padding_width = padding[1];
//   size_t dilation_height = dilation[0];
//   size_t dilation_width = dilation[1];

//   // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]

//   float *wdata = data_col->data;

  
//   for (int64_t in = 0; in < batch_size; in++) {
//     for (int64_t ioh = 0; ioh < out_height; ioh++) { // 1
//       for (int64_t iow = 0; iow < out_width; iow++) {
//         for (int64_t iic = ith; iic < IC; iic += nth) {

//           // micro kernel
//           float * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
//           const float * const src_data = (float *)((char *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

//           for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
//             for (int64_t ikw = 0; ikw < KW; ikw++) {
//               const int64_t iiw = iow*s0 + ikw*d0 - p0;
//               const int64_t iih = ioh*s1 + ikh*d1 - p1;

//               if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
//                 dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
//               } else {
//                 dst_data[iic*(KH*KW) + ikh*KW + ikw] = (src_data[iih*IW + iiw]);
//               }
//             }
//           }
//         }
//       }
//     }
//   }


//   size_t height_col = (im_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1;
//   size_t width_col = (im_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1;
//   size_t ksize = kernel_size[0];
//   size_t channels_col = im_channels * ksize * ksize;

//   for (size_t c = 0; c < channels_col; ++c) {
//     size_t w_offset = c % ksize;
//     size_t h_offset = (c / ksize) % ksize;
//     size_t c_im = c / ksize / ksize;
//     for (size_t h = 0; h < height_col; ++h) {
//       for (size_t w = 0; w < width_col; ++w) {
//         int h_pad = h * stride[0] + h_offset * dilation[0] - padding[0];
//         int w_pad = w * stride[1] + w_offset * dilation[1] - padding[1];
//         if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
//           ((float *)data_col->data)[(c * height_col + h) * width_col + w] =
//               ((float *)data_im->data)[(c_im * height + h_pad) * width + w_pad];
//         } else {
//           ((float *)data_col->data)[(c * height_col + h) * width_col + w] = 0;
//         }
//       }
//     }
//   }
// }
