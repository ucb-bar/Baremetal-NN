
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>

#include "nn.h"

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

static uint8_t floatEqual(float golden, float actual, float relErr) {
  return (fabs(actual - golden) < relErr) || (fabs((actual - golden) / actual) < relErr);
}

static uint8_t compareTensor(Tensor *golden, Tensor *actual) {
  for (size_t i = 0; i < golden->size; i += 1) {
    if (!floatEqual(((float *)golden->data)[i], ((float *)actual->data)[i], 1e-5)) {
      return 0;
    }
  }
  return 1;
}
