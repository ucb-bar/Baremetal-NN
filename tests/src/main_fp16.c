#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>

#include "nn.h"
// #include "riscv_vector.h"


void print_bits_half(float16_t x) {
  for(int i = 15; i >= 0; i -= 1) {
    printf("%d", ((x>>i)&1));
    if(i == 15 || i == 10)
      printf(" ");
    if(i == 10)
      printf("      ");
  }
  printf("\n");
}
void print_bits(float x) {
  uint32_t b = *(uint32_t*)&x;
  for(int i = 31; i >= 0; i -= 1) {
    printf("%d", ((b>>i)&1));
    if(i == 31 || i == 23)
      printf(" ");
    if(i == 23)
      printf("   ");
  }
  printf("\n");
}

uint8_t compareResult(float golden, float actual) {
  float diff = fabs(golden - actual);
  if (diff > 1e-2) {
    printf("FAILED ");
    printf("golden: ");
    NN_print_f32(golden, 6);
    printf("\n");
    printf("actual: ");
    NN_print_f32(actual, 6);
    printf("\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}

int main() {
  // for (size_t i = 0; i < 100; i += 1) {
    // float x = rand() / (float)RAND_MAX * 1000.0f;

    float x = (float)(0x47ca9334);

    float16_t x_compressed = NN_floatToHalf(x);
    float x_decompressed = NN_halfToFloat(x_compressed);
    
    print_bits(x);
    print_bits_half(x_compressed);
    print_bits(x_decompressed);

    printf("%f\t", x);
    printf("%f\n", x_decompressed);

    compareResult(x, x_decompressed);
  // }
  return 0;
}