#ifndef __NN_FLOAT16
#define __NN_FLOAT16

#include <stdint.h>
#include <stdlib.h>
#include <float.h>


#define float16_t uint16_t

// conversion method adapted from
// https://arxiv.org/abs/2112.08926
// and https://stackoverflow.com/a/60047308

/**
 * IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
 */
static inline float NN_halfToFloat(float16_t x) { 
  // exponent
  uint32_t e = (x & 0x7C00) >> 10;
  // mantissa
  uint32_t m = (x & 0x03FF) << 13;
  // evil log2 bit hack to count leading zeros in denormalized format
  uint32_t v = m >> 23;
  // sign : normalized : denormalized
  uint32_t result = ((x & 0x8000) << 16)
                  | ((e != 0) * ((e + 112) << 23 | m))
                  | (((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));
  return *((float *)(&result));
}

/**
 * IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
 */
static inline float16_t NN_floatToHalf(float x) {
  // round-to-nearest-even: add last bit after truncated mantissa
  uint32_t b = *((uint32_t *)(&x)) + 0x00001000;
  // exponent
  uint32_t e = (b & 0x7F800000)>>23;
  // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
  uint32_t m = b & 0x007FFFFF; 
  // sign : normalized : denormalized : saturate
  float16_t result = ((b & 0x80000000) >> 16)
                  | ((e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13))
                  | (((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) )
                  | (e > 143) * 0x7FFF; 
  return result;
}

#endif // __NN_FLOAT16