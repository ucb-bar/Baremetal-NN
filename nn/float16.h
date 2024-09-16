#ifndef __FLOAT16
#define __FLOAT16

#include <stdint.h>
#include <stdlib.h>
#include <float.h>

#ifdef X86
  #include <immintrin.h>
#endif


#ifdef FLT16_MAX
  typedef _Float16  float16_t;
#else
  typedef union {
    uint32_t i;
    float    f;
  } float_uint32_union_t;

  typedef uint16_t  float16_t;
#endif

/**
 * Converts a half-precision floating-point number to a single-precision floating-point number.
 * 
 * @param h The half-precision floating-point number to convert.
 * @return The single-precision floating-point number.
 */
static inline float NN_half_to_float(float16_t h) {
  #ifdef FLT16_MAX
    return (float)h;
  #else
    // from https://github.com/AcademySoftwareFoundation/Imath/blob/main/src/Imath/half.h
    // Note: This only supports the "round to even" rounding mode, which
    // was the only mode supported by the original OpenEXR library

    float_uint32_union_t v;
    // this code would be clearer, although it does appear to be faster
    // (1.06 vs 1.08 ns/call) to avoid the constants and just do 4
    // shifts.
    //
    uint32_t hexpmant = ((uint32_t) (h) << 17) >> 4;
    v.i               = ((uint32_t) (h >> 15)) << 31;

    // the likely really does help if most of your numbers are "normal" half numbers
    if ((hexpmant >= 0x00800000)) {
      v.i |= hexpmant;
      // either we are a normal number, in which case add in the bias difference
      // otherwise make sure all exponent bits are set
      if ((hexpmant < 0x0f800000)) {
        v.i += 0x38000000;
      }  
      else {
        v.i |= 0x7f800000;
      }
    }
    else if (hexpmant != 0) {
      // exponent is 0 because we're denormal, don't have to extract
      // the mantissa, can just use as is
      //
      // other compilers may provide count-leading-zeros primitives,
      // but we need the community to inform us of the variants
      uint32_t lc;
      lc = 0;
      while (0 == ((hexpmant << lc) & 0x80000000)) {
        lc += 1;
      }
      lc -= 8;
      // so nominally we want to remove that extra bit we shifted
      // up, but we are going to add that bit back in, then subtract
      // from it with the 0x38800000 - (lc << 23)....
      //
      // by combining, this allows us to skip the & operation (and
      // remove a constant)
      //
      // hexpmant &= ~0x00800000;
      v.i |= 0x38800000;
      // lc is now x, where the desired exponent is then
      // -14 - lc
      // + 127 -> new exponent
      v.i |= (hexpmant << lc);
      v.i -= (lc << 23);
    }
    return v.f;
  #endif
}


/**
 * Converts a single-precision floating-point number to a half-precision floating-point number.
 * 
 * @param f The single-precision floating-point number to convert.
 * @return The half-precision floating-point number.
 */
static inline float16_t NN_float_to_half(float f) {
  #ifdef FLT16_MAX
    return (_Float16)f;
  #else
    // from https://github.com/AcademySoftwareFoundation/Imath/blob/main/src/Imath/half.h
    // Note: This only supports the "round to even" rounding mode, which
    // was the only mode supported by the original OpenEXR library
    
    float_uint32_union_t  v;
    float16_t ret;
    uint32_t e, m, ui, r, shift;

    v.f = f;

    ui  = (v.i & ~0x80000000);
    ret = ((v.i >> 16) & 0x8000);

    // exponent large enough to result in a normal number, round and return
    if (ui >= 0x38800000) {
      // inf or nan
      if (ui >= 0x7f800000) {
        ret |= 0x7c00;
        if (ui == 0x7f800000) {
          return ret;
        }
        m = (ui & 0x7fffff) >> 13;
        // make sure we have at least one bit after shift to preserve nan-ness
        return ret | (uint16_t) m | (uint16_t) (m == 0);
      }

      // too large, round to infinity
      if (ui > 0x477fefff) {
        return ret | 0x7c00;
      }

      ui -= 0x38000000;
      ui = ((ui + 0x00000fff + ((ui >> 13) & 1)) >> 13);
      return ret | (uint16_t) ui;
    }

    // zero or flush to 0
    if (ui < 0x33000001) {
      return ret;
    }

    // produce a denormalized half
    e     = (ui >> 23);
    shift = 0x7e - e;
    m     = 0x800000 | (ui & 0x7fffff);
    r     = m << (32 - shift);
    ret  |= (m >> shift);
    if (r > 0x80000000 || (r == 0x80000000 && (ret & 0x1) != 0)) {
      ret += 1;
    }
    return ret;
  #endif
}

#endif // __FLOAT16
