
#include "nn_linear.h"

void NN_linear_F32(Tensor *y, Tensor *x, Tensor *w, Tensor *b) {
  // assert(x->dtype == DTYPE_F32);
  // assert(w->dtype == DTYPE_F32);
  // assert(b->dtype == DTYPE_F32);
  // assert(x->shape[1] == w->shape[1]);
  // assert(b->shape[1] == w->shape[0]);

  // y->dtype = DTYPE_F32;
  // y->shape[0] = x->shape[0];   // batch size
  // y->shape[1] == b->shape[1];  // out_features

  // float *y_data = (float *)y->data;
  // float *x_data = (float *)x->data;
  // float *w_data = (float *)w->data;
  // float *b_data = (float *)b->data;
  
  // for (size_t b = 0; b < y->shape[0]; b += 1) {
  //   for (size_t i = 0; i < w->shape[0]; i += 1) {
  //     y_data[b * y->shape[1] + i] = 0;
  //     for (size_t j = 0; j < w->shape[1]; j += 1) {
  //       y_data[b * y->shape[1] + i] += x_data[b * x->shape[1] + j] * w_data[i * w->shape[1] + j];
  //     }
  //   }
  // }

  NN_transpose_F32(w, w);
  NN_matmul_F32(y, x, w);
  NN_add_F32(y, y, b);
}
