
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>

#include "nn.h"

#ifdef X86
  #include <immintrin.h>
  #include <x86intrin.h>
#endif

#ifdef RVV
  #include "riscv_vector.h"
#endif

static void enableAcceleratorFeatures() {
  #ifdef RVV
    // enable vector operation
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x00000600 | 0x00006000 | 0x00018000;
    asm volatile("csrw mstatus, %0"::"r"(mstatus));
  #endif
}

static size_t readCycles() {
  #ifdef X86
    return __rdtsc();
  #elif defined(RVV)
    return READ_CSR("mcycle");
  #endif
}

static uint8_t floatEqual(float golden, float actual, float rel_err) {
  return (fabs(actual - golden) < rel_err) || (fabs((actual - golden) / actual) < rel_err);
}

static uint8_t compareTensor(Tensor *golden, Tensor *actual, float rel_err) {
  switch (golden->dtype) {
    case DTYPE_F16:
      for (size_t i = 0; i < golden->size; i += 1) {
        if (!floatEqual(NN_halfToFloat(((float16_t *)golden->data)[i]), NN_halfToFloat(((float16_t *)actual->data)[i]), rel_err)) {
          return 0;
        }
      }
      return 1;
    case DTYPE_F32:
      for (size_t i = 0; i < golden->size; i += 1) {
        if (!floatEqual(((float *)golden->data)[i], ((float *)actual->data)[i], rel_err)) {
          return 0;
        }
      }
      return 1;
    default:
      return 0;
  }
}
