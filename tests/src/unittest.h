
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <riscv.h>

#include "nn.h"

#ifdef X86
  #include <immintrin.h>
  #include <x86intrin.h>
#endif

static void enable_accelerator_features() {
  #ifdef CONFIG_BACKEND_RISCV_V
    // enable vector operation
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x00000400 | 0x00004000 | 0x00010000;
    asm volatile("csrw mstatus, %0"::"r"(mstatus));
  #endif
}

static size_t read_cycles() {
  #if CONFIG_TOOLCHAIN_RISCV
    return READ_CSR("mcycle");
  #endif
  return 0;
}
