
#include "nn_im2col.h"

// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
void ggml_compute_forward_im2col_f32(Tensor *out, Tensor *kernel, Tensor *image) {
  const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
  const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
  const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
  const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
  const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
  const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
  const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t N  = is_2D ? ne13 : ne12;
  const int64_t IC = is_2D ? ne12 : ne11;
  const int64_t IH = is_2D ? ne11 : 1;
  const int64_t IW = ne10;

  const int64_t KH = is_2D ? ne01 : 1;
  const int64_t KW = ne00;

  const int64_t OH = is_2D ? ne2 : 1;
  const int64_t OW = ne1;

  int ofs0 = is_2D ? nb13 : nb12;
  int ofs1 = is_2D ? nb12 : nb11;

  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb10 == sizeof(float));

  if (params->type == GGML_TASK_TYPE_INIT) {
      return;
  }

  if (params->type == GGML_TASK_TYPE_FINALIZE) {
      return;
  }

  // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
  {
      float * const wdata = (float *) dst->data;

      for (int64_t in = 0; in < N; in++) {
          for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
              for (int64_t iow = 0; iow < OW; iow++) {
                  for (int64_t iic = ith; iic < IC; iic += nth) {

                      // micro kernel
                      float * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                      const float * const src_data = (float *)((char *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                      for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                          for (int64_t ikw = 0; ikw < KW; ikw++) {
                              const int64_t iiw = iow*s0 + ikw*d0 - p0;
                              const int64_t iih = ioh*s1 + ikh*d1 - p1;

                              if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                  dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                              } else {
                                  dst_data[iic*(KH*KW) + ikh*KW + ikw] = (src_data[iih*IW + iiw]);
                              }
                          }
                      }
                  }
              }
          }
      }
  }
}
