
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <riscv.h>

#include "nn.h"

#ifdef X86
  #include <immintrin.h>
  #include <x86intrin.h>
#endif

#ifdef RISCV_V
  #include "riscv_vector.h"
#endif

static void enable_accelerator_features() {
  #ifdef RISCV_V
    // enable vector operation
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x00000400 | 0x00004000 | 0x00010000;
    asm volatile("csrw mstatus, %0"::"r"(mstatus));
  #endif
}

static size_t read_cycles() {
  #ifdef X86
    return __rdtsc();
  #elif defined(RISCV)
    return READ_CSR("mcycle");
  #endif
}
