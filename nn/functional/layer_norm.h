#ifndef __NN_LAYER_NORM_H
#define __NN_LAYER_NORM_H

#include <assert.h>
#include <math.h>

#include "tensor.h"
#include "ops/sum.h"
#include "ops/add.h"
#include "ops/add1.h"
#include "ops/mul.h"
#include "ops/mul1.h"
#include "ops/sqr.h"


void NN_layer_norm(
  Tensor *out, const Tensor *in,
  size_t normalized_dims,
  const Tensor *weight, const Tensor *bias,
  const float eps);


#endif // __NN_LAYER_NORM_H
