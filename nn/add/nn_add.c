
#include "nn_add.h"


void NN_add(Tensor *out, Tensor *a, Tensor *b) {
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32) {
    NN_add_F32(out, a, b);
    return;
  }
  if ((a->dtype == DTYPE_I8 || a->dtype == DTYPE_I32) && (b->dtype == DTYPE_I8 || b->dtype == DTYPE_I32)) {
    NN_add_INT(out, a, b);
    return;
  }

  printf("Unsupported operation: %s + %s -> %s\n", NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype), NN_getDataTypeName(out->dtype));
}

void NN_add_F32(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(a->shape[0] == b->shape[0]);
  
  out->dtype = DTYPE_F32;
  out->shape[0] = a->shape[0];
  out->shape[1] = a->shape[1];
  
  float *out_data = (float *)out->data;
  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;

  if (b->ndim == 1 || b->shape[0] == 1) {
    printf("Info: performing ADD with broadcasting\n");

    for (size_t i = 0; i<out->shape[0]; i+=1) {
      for (size_t j = 0; j<out->shape[1]; j+=1) {
        out_data[i*out->shape[1]+j] = a_data[i*a->shape[1]+j] + b_data[j];
      }
    }
    return;
  }

  if (b->ndim == 2) {
    assert(a->shape[1] == b->shape[1]);

    for (size_t i = 0; i<out->size; i+=1) {
      out_data[i] = a_data[i] + b_data[i];
    }
    return;
  }

  printf("Unsupported operation between tensor with shape ");
  NN_printShape(a->shape);
  printf(" and ");
  NN_printShape(b->shape);
  printf("\n");
}

void NN_add_INT(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->dtype == DTYPE_I8 || a->dtype == DTYPE_I32);
  assert(b->dtype == DTYPE_I8 || b->dtype == DTYPE_I32);
  assert(a->shape[0] == b->shape[0]);

  out->shape[0] = a->shape[0];
  out->shape[1] = a->shape[1];
  
  if (b->ndim == 1 || b->shape[0] == 1) {
    printf("Info: performing ADD with broadcasting\n");

    for (size_t i = 0; i<out->shape[0]; i+=1) {
      for (size_t j = 0; j<out->shape[1]; j+=1) {
        int32_t a_val, b_val, out_val;

        if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8) {
          out->dtype = DTYPE_I8;
          a_val = ((int8_t *)a->data)[i*a->shape[1]+j];
          b_val = ((int8_t *)b->data)[j];
        }
        else {
          out->dtype = DTYPE_I32;
          if (a->dtype == DTYPE_I32) {
            a_val = ((int32_t *)a->data)[i*a->shape[1]+j];
          }
          else {
            a_val = ((int8_t *)a->data)[i*a->shape[1]+j];
          }
          if (b->dtype == DTYPE_I32) {
            b_val = ((int32_t *)b->data)[j];
          }
          else {
            b_val = ((int8_t *)b->data)[j];
          }
        }
        out_val = a_val + b_val;
        
        if (out->dtype == DTYPE_I8) {
          ((int8_t *)out->data)[i*out->shape[1]+j] = (int8_t)out_val;
        }
        else {
          ((int32_t *)out->data)[i*out->shape[1]+j] = out_val;
        }
      }
    }
    return;
  }

  if (b->ndim == 2) {
    assert(a->shape[1] == b->shape[1]);
    
    for (size_t i = 0; i<out->size; i+=1) {
      int32_t a_val, b_val, out_val;

      if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8) {
        out->dtype = DTYPE_I8;
        a_val = ((int8_t *)a->data)[i];
        b_val = ((int8_t *)b->data)[i];
      }
      else {
        out->dtype = DTYPE_I32;
        if (a->dtype == DTYPE_I32) {
          a_val = ((int32_t *)a->data)[i];
        }
        else {
          a_val = ((int8_t *)a->data)[i];
        }
        if (b->dtype == DTYPE_I32) {
          b_val = ((int32_t *)b->data)[i];
        }
        else {
          b_val = ((int8_t *)b->data)[i];
        }
      }
      out_val = a_val + b_val;
      
      if (out->dtype == DTYPE_I8) {
        ((int8_t *)out->data)[i] = (int8_t)out_val;
      }
      else {
        ((int32_t *)out->data)[i] = out_val;
      }
    }
  }

  printf("Unsupported operation between tensor with shape ");
  NN_printShape(a->shape);
  printf(" and ");
  NN_printShape(b->shape);
  printf("\n");
}

