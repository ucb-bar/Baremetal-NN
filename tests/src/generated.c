
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>

#include "nn.h"
#include "unittest.h"

int main() {
  enableAcceleratorFeatures();

  size_t cycles = 0;

  
  {
    printf("abs:                    ");

    // -1.66, -2.27, 3.07, 0.9375
    Tensor *a = NN_tensor(2, (size_t[]){ 1, 4 }, DTYPE_F16, (uint8_t[]){ 0xa4, 0xbe, 0x8a, 0xc0, 0x24, 0x42, 0x80, 0x3b });


    // 1.66, 2.27, 3.07, 0.9375
    Tensor *golden = NN_tensor(2, (size_t[]){ 1, 4 }, DTYPE_F16, (uint8_t[]){ 0xa4, 0x3e, 0x8a, 0x40, 0x24, 0x42, 0x80, 0x3b });
    Tensor *actual = NN_zeros(2, (size_t[]){ 1, 4 }, DTYPE_F16);
    cycles = readCycles();
    NN_abs(actual, a);
    cycles = readCycles() - cycles;
    printf("%s  (%lu cycles)\n", compareTensor(golden, actual, 1e-3) ? "PASS" : "FAIL", cycles);


    NN_deleteTensor(a);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

}