
#include "nn_add.h"


void NN_add(Tensor *out, Tensor *a, Tensor *b) {
  assert(b->ndim <= a->ndim);

  switch (b->dtype) {
    case DTYPE_F32:
      if (b->ndim == a->ndim) {
        if (b->shape[0] == a->shape[0]
            && b->shape[1] == a->shape[1]
            && b->shape[2] == a->shape[2]
            && b->shape[3] == a->shape[3]) {
          NN__add_F32(out->size, (float *)out->data, (float *)a->data, (float *)b->data);
          return;
        }
        for (size_t i = 0; i < out->shape[0]; i += 1) {
          for (size_t j = 0; j < out->shape[1]; j += 1) {
            // handle broadcasting
            size_t a_i = i < a->shape[0] ? i : 0;
            size_t a_j = j < a->shape[1] ? j : 0;
            size_t b_i = i < b->shape[0] ? i : 0;
            size_t b_j = j < b->shape[1] ? j : 0;

            ((float *)out->data)[i * out->shape[1] + j]
              = ((float *)a->data)[a_i * a->shape[1] + a_j]
              + ((float *)b->data)[b_i * b->shape[1] + b_j];
          }
        }
        return;
      }
      // broadcast
      if (b->ndim == 1 && a->ndim == 2) {
        if (b->shape[0] == a->shape[1]) {
          for (size_t i = 0; i < out->shape[0]; i += 1) {
            for (size_t j = 0; j < out->shape[1]; j += 1) {
              // handle broadcasting
              size_t a_i = i < a->shape[0] ? i : 0;
              size_t a_j = j < a->shape[1] ? j : 0;

              ((float *)out->data)[i * out->shape[1] + j]
                = ((float *)a->data)[a_i * a->shape[1] + a_j]
                + ((float *)b->data)[j];
            }
          }
          return;
        }
      }
      
      printf("[ERROR] Unsupported operation between tensor with shape ");
      NN_printShape(a);
      printf(" + ");
      NN_printShape(b);
      printf("\n");
      return;

    default:
      break;
  }

  printf("[ERROR] Unsupported operation between tensor with dtype %s = %s + %s\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype)
  );
}

void NN_add1(Tensor *out, Tensor *a, float b) {
  assert(out->ndim == a->ndim);
  assert(out->dtype == a->dtype);
  assert(out->size == a->size);

  switch (out->dtype) {
    case DTYPE_F32:
      NN__add1_F32(out->size, (float *)out->data, (float *)a->data, b);
      return;

    default:
      break;
  }
  printf("[ERROR] Unsupported operation between tensor with dtype %s += %s\n", 
    NN_getDataTypeName(out->dtype), NN_getDataTypeName(a->dtype)
  );
}

void NN_addInplace(Tensor *b, Tensor *a) {
  assert(b->ndim == a->ndim);
  assert(b->dtype == a->dtype);

  switch (b->dtype) {
    case DTYPE_F32:
      NN__acc_F32(b->size, (float *)b->data, (float *)a->data);
      return;
    case DTYPE_I8:
      NN__acc_I8(b->size, (int8_t *)b->data, (int8_t *)a->data);
      return;
    default:
      break;
  }

  printf("[ERROR] Unsupported operation between tensor with dtype %s += %s\n", 
    NN_getDataTypeName(b->dtype), NN_getDataTypeName(a->dtype)
  );
}

void NN_addInplace1(Tensor *b, float scalar) {
  switch (b->dtype) {
    case DTYPE_F32:
      NN__acc1_F32(b->size, (float *)b->data, scalar);
      return;
    default:
      break;
  }

  printf("[ERROR] Unsupported operation between tensor with dtype %s += float\n", 
    NN_getDataTypeName(b->dtype)
  );
}
