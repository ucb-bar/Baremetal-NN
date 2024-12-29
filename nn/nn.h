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


#include "nn_f16.h"

#include "nn_f32.h"


#endif // __NN_H