
#include "nn_add.h"


void NN_add(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->shape[0] == b->shape[0]);
  assert(a->shape[1] == b->shape[1]);
  
  if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32) {
    NN_add_F32(out, a, b);
    return;
  }
  if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8 && out->dtype == DTYPE_I8) {
    NN_add_I8(out, a, b);
    return;
  }
  if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I8 && out->dtype == DTYPE_I32) {
    NN_add_I8_I8_I32(out, a, b);
    return;
  }
  if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I32 && out->dtype == DTYPE_I32) {
    NN_add_I32(out, a, b);
    return;
  }
  if (a->dtype == DTYPE_I32 && b->dtype == DTYPE_I8 && out->dtype == DTYPE_I32) {
    NN_add_I32_I8_I32(out, a, b);
    return;
  }
  if (a->dtype == DTYPE_I8 && b->dtype == DTYPE_I32 && out->dtype == DTYPE_I32) {
    NN_add_I32_I8_I32(out, b, a);
    return;
  }

  printf("Unsupported operation: %s + %s -> %s\n", NN_getDataTypeName(a->dtype), NN_getDataTypeName(b->dtype), NN_getDataTypeName(out->dtype));
}

void NN_add_I8(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->shape[0] == b->shape[0]);
  assert(a->dtype == DTYPE_I8);
  assert(b->dtype == DTYPE_I8);
  assert(out->dtype == DTYPE_I8);

  if (b->ndim == 0) {
    printf("Warning: broadcasting support will be removed in the future\n");
    for (size_t i = 0; i<out->size; i+=1) {
      ((int8_t *)out->data)[i] = (int8_t)((int8_t *)a->data)[i] + (int8_t)((int8_t *)b->data)[0];
    }
    return;
  }

  if (b->ndim == 1) {
    assert(a->shape[1] == b->shape[0]);
    printf("Warning: broadcasting support will be removed in the future\n");

    for (size_t i = 0; i<out->shape[0]; i+=1) {
      for (size_t j = 0; j<out->shape[1]; j+=1) {
        ((int8_t *)out->data)[i*out->shape[1]+j] = (int8_t)((int8_t *)a->data)[i*a->shape[1]+j] + (int8_t)((int8_t *)b->data)[j];
      }
    }
    return;
  }

  if (b->ndim == 2) {
    assert(a->shape[1] == b->shape[1]);

    for (size_t i = 0; i<out->size; i+=1) {
      ((int8_t *)out->data)[i] = (int8_t)((int8_t *)a->data)[i] + (int8_t)((int8_t *)b->data)[i];
    }
    return;
  }

  printf("Unsupported operation between dimensions: %zu and %zu\n", a->ndim, b->ndim);
}

void NN_add_I8_I8_I32(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->shape[0] == b->shape[0]);
  assert(a->dtype == DTYPE_I8);
  assert(b->dtype == DTYPE_I8);
  assert(out->dtype == DTYPE_I32);

  if (b->ndim == 0) {
    printf("Warning: broadcasting support will be removed in the future\n");
    for (size_t i = 0; i<out->size; i+=1) {
      ((int32_t *)out->data)[i] = (int32_t)((int8_t *)a->data)[i] + (int32_t)((int8_t *)b->data)[0];
    }
    return;
  }

  if (b->ndim == 1) {
    assert(a->shape[1] == b->shape[0]);
    printf("Warning: broadcasting support will be removed in the future\n");

    for (size_t i = 0; i<out->shape[0]; i+=1) {
      for (size_t j = 0; j<out->shape[1]; j+=1) {
        ((int32_t *)out->data)[i*out->shape[1]+j] = (int32_t)((int8_t *)a->data)[i*a->shape[1]+j] + (int32_t)((int8_t *)b->data)[j];
      }
    }
    return;
  }

  if (b->ndim == 2) {
    assert(a->shape[1] == b->shape[1]);

    for (size_t i = 0; i<out->size; i+=1) {
      ((int32_t *)out->data)[i] = (int32_t)((int8_t *)a->data)[i] + (int32_t)((int8_t *)b->data)[i];
    }
    return;
  }

  printf("Unsupported operation between dimensions: %zu and %zu\n", a->ndim, b->ndim);
}


void NN_add_I32_I8_I32(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->shape[0] == b->shape[0]);
  assert(a->dtype == DTYPE_I32);
  assert(b->dtype == DTYPE_I8);
  assert(out->dtype == DTYPE_I32);

  if (b->ndim == 0) {
    printf("Warning: broadcasting support will be removed in the future\n");
    for (size_t i = 0; i<out->size; i+=1) {
      ((int32_t *)out->data)[i] = ((int32_t *)a->data)[i] + (int32_t)((int8_t *)b->data)[0];
    }
    return;
  }

  if (b->ndim == 1) {
    assert(a->shape[1] == b->shape[0]);
    printf("Warning: broadcasting support will be removed in the future\n");

    for (size_t i = 0; i<out->shape[0]; i+=1) {
      for (size_t j = 0; j<out->shape[1]; j+=1) {
        ((int32_t *)out->data)[i*out->shape[1]+j] = ((int32_t *)a->data)[i*a->shape[1]+j] + (int32_t)((int8_t *)b->data)[j];
      }
    }
    return;
  }

  if (b->ndim == 2) {
    assert(a->shape[1] == b->shape[1]);

    for (size_t i = 0; i<out->size; i+=1) {
      ((int32_t *)out->data)[i] = ((int32_t *)a->data)[i] + (int32_t)((int8_t *)b->data)[i];
    }
    return;
  }

  printf("Unsupported operation between dimensions: %zu and %zu\n", a->ndim, b->ndim);
}

void NN_add_I32(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->shape[0] == b->shape[0]);
  assert(a->dtype == DTYPE_I32);
  assert(b->dtype == DTYPE_I32);
  assert(out->dtype == DTYPE_I32);

  if (b->ndim == 0) {
    printf("Warning: broadcasting support will be removed in the future\n");
    for (size_t i = 0; i<out->size; i+=1) {
      ((int32_t *)out->data)[i] = ((int32_t *)a->data)[i] + ((int32_t *)b->data)[0];
    }
    return;
  }

  if (b->ndim == 1) {
    assert(a->shape[1] == b->shape[0]);
    printf("Warning: broadcasting support will be removed in the future\n");

    for (size_t i = 0; i<out->shape[0]; i+=1) {
      for (size_t j = 0; j<out->shape[1]; j+=1) {
        ((int32_t *)out->data)[i*out->shape[1]+j] = ((int32_t *)a->data)[i*a->shape[1]+j] + ((int32_t *)b->data)[j];
      }
    }
    return;
  }

  if (b->ndim == 2) {
    assert(a->shape[1] == b->shape[1]);

    for (size_t i = 0; i<out->size; i+=1) {
      ((int32_t *)out->data)[i] = ((int32_t *)a->data)[i] + ((int32_t *)b->data)[i];
    }
    return;
  }
  
  printf("Unsupported operation between dimensions: %zu and %zu\n", a->ndim, b->ndim);
}

void NN_add_F32(Tensor *out, Tensor *a, Tensor *b) {
  assert(a->shape[0] == b->shape[0]);
  assert(a->dtype == DTYPE_F32);
  assert(b->dtype == DTYPE_F32);
  assert(out->dtype == DTYPE_F32);
    printf("Warning: broadcasting support will be removed in the future\n");

  if (b->ndim == 0) {
    for (size_t i = 0; i<out->size; i+=1) {
      ((float *)out->data)[i] = ((float *)a->data)[i] + ((float *)b->data)[0];
    }
    return;
  }

  if (b->ndim == 1) {
    assert(a->shape[1] == b->shape[0]);
    printf("Warning: broadcasting support will be removed in the future\n");

    for (size_t i = 0; i<out->shape[0]; i+=1) {
      for (size_t j = 0; j<out->shape[1]; j+=1) {
        ((float *)out->data)[i*out->shape[1]+j] = ((float *)a->data)[i*a->shape[1]+j] + ((float *)b->data)[j];
      }
    }
    return;
  }

  if (b->ndim == 2) {
    assert(a->shape[1] == b->shape[1]);

    for (size_t i = 0; i<out->size; i+=1) {
      ((float *)out->data)[i] = ((float *)a->data)[i] + ((float *)b->data)[i];
    }
    return;
  }

  printf("Unsupported operation between dimensions: %zu and %zu\n", a->ndim, b->ndim);
}
