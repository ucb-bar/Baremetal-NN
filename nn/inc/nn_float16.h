#ifndef __NN_FLOAT16
#define __NN_FLOAT16

#include <stdint.h>
#include <stdlib.h>
#include <float.h>


typedef uint16_t float16_t;

typedef union {
  uint32_t i;
  float    f;
} float_uint32_union_t;


// from https://github.com/AcademySoftwareFoundation/Imath/blob/main/src/Imath/half.h

static inline float NN_halfToFloat(float16_t h) {
  #if defined(__F16C__)
    // NB: The intel implementation does seem to treat NaN slightly
    // different than the original toFloat table does (i.e. where the
    // 1 bits are, meaning the signalling or not bits). This seems
    // benign, given that the original library didn't really deal with
    // signalling vs non-signalling NaNs
    #ifdef _MSC_VER
      /* msvc does not seem to have cvtsh_ss :( */
      return _mm_cvtss_f32 (_mm_cvtph_ps (_mm_set1_epi16 (h)));
    #else
      return _cvtsh_ss (h);
    #endif
  #else
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
      //
      // other compilers may provide count-leading-zeros primitives,
      // but we need the community to inform us of the variants
      uint32_t lc;
      #if defined(_MSC_VER)
        // The direct intrinsic for this is __lznct, but that is not supported
        // on older x86_64 hardware or ARM. Instead uses the bsr instruction
        // and one additional subtraction. This assumes hexpmant != 0, for 0
        // bsr and lznct would behave differently.
        unsigned long bsr;
        _BitScanReverse(&bsr, hexpmant);
        lc = (31 - bsr);
      #elif defined(__GNUC__) || defined(__clang__)
        lc = (uint32_t)__builtin_clz(hexpmant);
      #else
        lc = 0;
        while (0 == ((hexpmant << lc) & 0x80000000))
          lc += 1;
      #endif
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

///
/// Convert half to float
///
/// Note: This only supports the "round to even" rounding mode, which
/// was the only mode supported by the original OpenEXR library
///

static inline float16_t NN_floatToHalf(float f) {
  #if defined(__F16C__)
    // #ifdef _MSC_VER
    //   // msvc does not seem to have cvtsh_ss :(
    //   return _mm_extract_epi16 (
    //       _mm_cvtps_ph (
    //           _mm_set_ss (f), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
    //       0);
    // #else
    //   // preserve the fixed rounding mode to nearest
    //   return _cvtss_sh (f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    // #endif
  #else
    float_uint32_union_t  v;
    float16_t ret;
    uint32_t e, m, ui, r, shift;

    v.f = f;

    ui  = (v.i & ~0x80000000);
    ret = ((v.i >> 16) & 0x8000);

    // exponent large enough to result in a normal number, round and return
    if (ui >= 0x38800000)
    {
        // inf or nan
        if (ui >= 0x7f800000) {
            ret |= 0x7c00;
            if (ui == 0x7f800000) return ret;
            m = (ui & 0x7fffff) >> 13;
            // make sure we have at least one bit after shift to preserve nan-ness
            return ret | (uint16_t) m | (uint16_t) (m == 0);
        }

        // too large, round to infinity
        if (ui > 0x477fefff) {
#    ifdef IMATH_HALF_ENABLE_FP_EXCEPTIONS
            feraiseexcept (FE_OVERFLOW);
#    endif
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
    ret |= (m >> shift);
    if (r > 0x80000000 || (r == 0x80000000 && (ret & 0x1) != 0)) ++ret;
    return ret;
#endif
}

#endif // __NN_FLOAT16