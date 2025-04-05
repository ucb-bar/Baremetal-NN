/**
 * @author AlexIII
 * @brief Arduino wrapper for incbin.h
 */

#ifndef _ARDUINO_INCBIN_LIB_H_
#define _ARDUINO_INCBIN_LIB_H_

#ifndef INCBIN_OUTPUT_SECTION
#   if defined(__AVR__)
#       define INCBIN_OUTPUT_SECTION ".text.progmem"
#       define INCBIN_SIZE_OUTPUT_SECTION ".rodata"
#   elif defined(ESP8266)
#       define INCBIN_OUTPUT_SECTION ".irom.text"
#       define INCBIN_SIZE_OUTPUT_SECTION ".rodata"
#   endif
#endif

#define INCBIN_APPEND_TERMINATING_NULL

#include "_incbin.h"    // include the original library

#if defined(__AVR__) || defined(ESP8266)
#   define INCTEXT(NAME, FILENAME) \
        INCBIN_PTR_TYPE(NAME, FILENAME, __FlashStringHelper)
#else
#   define INCTEXT(NAME, FILENAME) \
        INCBIN_PTR_TYPE(NAME, FILENAME, char)
#endif

#ifndef INCTXT
#   define INCTXT(NAME, FILENAME) INCTEXT(NAME, FILENAME)
#endif

#endif /* _ARDUINO_INCBIN_LIB_H_ */
