#ifndef __NN_LAYER_NORM_H
#define __NN_LAYER_NORM_H

#include <assert.h>
#include <math.h>

#include "nn_tensor.h"


void NN_layer_norm(
  Tensor *out, Tensor *in,
  Tensor *weight, Tensor *bias,
  const float eps);


#endif // __NN_LAYER_NORM_H
