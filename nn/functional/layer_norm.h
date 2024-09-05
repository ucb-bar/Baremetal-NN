#ifndef __NN_LAYER_NORM_H
#define __NN_LAYER_NORM_H

#include <assert.h>
#include <math.h>

#include "tensor.h"
#include "kernel/sum.h"
#include "kernel/add.h"
#include "kernel/add1.h"
#include "kernel/mul.h"
#include "kernel/mul1.h"
#include "kernel/sqr.h"


void NN_layer_norm(
  Tensor *out, const Tensor *in,
  size_t normalized_dims,
  const Tensor *weight, const Tensor *bias,
  const float eps);


#endif // __NN_LAYER_NORM_H
