
#include "nn_norm.h"


void NN_norm(Tensor *result, const Tensor *x) {
  assert(x->ndim == 2);
  assert(NN_is_scalar(result));
  assert(result->dtype == x->dtype);

  float sum = 0;

  switch (x->dtype) {
    case DTYPE_F32:
      for (size_t i = 0; i < x->shape[0]; i += 1) {
        for (size_t j = 0; j < x->shape[1]; j += 1) {
          sum += pow(((float *)x->data)[i * x->shape[1] + j], 2);
        }
      }
      return;

    default:
      break;
  }

  ((float *)result->data)[0] = sqrt(sum);
  
  printf("[ERROR] Unsupported operation between tensor with dtype %s = ||%s||\n", 
    NN_get_datatype_name(result->dtype), NN_get_datatype_name(x->dtype)
  );
}
