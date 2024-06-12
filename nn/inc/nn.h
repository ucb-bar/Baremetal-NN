#ifndef __NN_H
#define __NN_H

#include <assert.h>

#include "nn_float16.h"
#include "nn_tensor.h"
#include "nn_print.h"
#include "nn_abs.h"
#include "nn_add.h"
#include "nn_batchnorm2d.h"
#include "nn_conv2d.h"
#include "nn_copy.h"
#include "nn_interpolate.h"
#include "nn_linear.h"
#include "nn_matmul.h"
#include "nn_matrixnorm.h"
#include "nn_max.h"
#include "nn_min.h"
#include "nn_mul.h"
#include "nn_neg.h"
#include "nn_relu.h"
#include "nn_relu6.h"
#include "nn_sub.h"
#include "nn_sum.h"


// http://elm-chan.org/junk/32bit/binclude.html
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



void NN_assert(int condition, char *message) {
  if (!condition) {
    printf("Assertion failed: ");
    printf("%s\n", message);
    exit(1);
  }
}



#endif // __NN_H