/**
 * @file nn.h
 * @brief The Baremetal-NN Library
 * 
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

#ifndef __NN_H
#define __NN_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


// http://elm-chan.org/junk/32bit/binclude.html
#ifdef __APPLE__
#define INCLUDE_FILE(section, filename, symbol) asm (\
    ".align 4\n"                             /* Word alignment */\
    ".globl _"#symbol"_start\n"              /* Export the object start address */\
    ".globl _"#symbol"_data\n"               /* Export the object address */\
    "_"#symbol"_start:\n"                    /* Define the object start address label */\
    "_"#symbol"_data:\n"                     /* Define the object label */\
    ".incbin \""filename"\"\n"               /* Import the file */\
    ".globl _"#symbol"_end\n"                /* Export the object end address */\
    "_"#symbol"_end:\n"                      /* Define the object end address label */\
    ".align 4\n")                            /* Word alignment */
#else
#define INCLUDE_FILE(section, filename, symbol) asm (\
    ".section "#section"\n"                   /* Change section */\
    ".balign 4\n"                             /* Word alignment */\
    ".global "#symbol"_start\n"               /* Export the object start address */\
    ".global "#symbol"_data\n"                /* Export the object address */\
    #symbol"_start:\n"                        /* Define the object start address label */\
    #symbol"_data:\n"                         /* Define the object label */\
    ".incbin \""filename"\"\n"                /* Import the file */\
    ".global "#symbol"_end\n"                 /* Export the object end address */\
    #symbol"_end:\n"                          /* Define the object end address label */\
    ".balign 4\n"                             /* Word alignment */\
    ".section \".text\"\n")                   /* Restore section */
#endif

/**
 * nn_assert
 * 
 * Asserts that a condition is true. If the condition is false, it prints an error message and exits.
 * 
 * @param condition The condition to assert.
 * @param message The error message to print if the condition is false.
 */
static inline void nn_assert(int condition, char *message) {
  if (!condition) {
    printf("Assertion failed: ");
    printf("%s\n", message);
    exit(1);
  }
}


/**
 * nn_print_shape
 * 
 * Prints the shape of the tensor.
 * 
 * @param ndim The number of dimensions.
 * @param shape The shape to print.
 */
void nn_print_shape(size_t ndim, const size_t *shape) {
  printf("(");
  for (size_t i = 0; i < ndim; i += 1) {
    printf("%d", (int)shape[i]);
    if (i < ndim-1) {
      printf(", ");
    }
  }
  printf(")");
}


/**
 * nn_print_f32
 * 
 * Prints a float.
 * 
 * @param v The float to print.
 * @param num_digits The number of decimal digits to print.
 */
void nn_print_f32(float v, int16_t num_digits) {
  if (isinf(v)) {
    if (signbit(v)) {
      printf("-inf");
    } else {
      printf("inf");
    }
    return;
  }
  
  if (v < 0) {
    printf("-");  // Print the minus sign for negative numbers
    v = -v;        // Make the number positive for processing
  }

  // Calculate the integer part of the number
  long int_part = (long)v;
  float fractional_part = v - int_part;

  // Print the integer part
  printf("%ld", int_part);

  if (num_digits > 0) {
    printf("."); // Print the decimal point
  }

  // Handle the fractional part
  while (num_digits > 0) {
    num_digits -= 1;
    fractional_part *= 10;
    int digit = (int)(fractional_part);
    printf("%d", digit);
    fractional_part -= digit;
  }
}


#include "nn_f16.h"

#include "nn_f32.h"


#endif // __NN_H