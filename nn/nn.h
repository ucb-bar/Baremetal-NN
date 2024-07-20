#ifndef __NN_H
#define __NN_H

#include <assert.h>

#include "nn_float16.h"
#include "nn_tensor.h"
#include "functional/nn_tensor_creation.h"
#include "functional/nn_print.h"
#include "functional/nn_abs.h"
#include "functional/nn_add.h"
#include "functional/nn_batch_norm2d.h"
#include "functional/nn_clip.h"
#include "functional/nn_conv2d.h"
#include "functional/nn_copy.h"
#include "functional/nn_div.h"
#include "functional/nn_elu.h"
#include "functional/nn_fill.h"
#include "functional/nn_interpolate.h"
#include "functional/nn_layer_norm.h"
#include "functional/nn_linear.h"
#include "functional/nn_matmul.h"
#include "functional/nn_norm.h"
#include "functional/nn_max.h"
#include "functional/nn_mm.h"
#include "functional/nn_maximum.h"
#include "functional/nn_min.h"
#include "functional/nn_minimum.h"
#include "functional/nn_mul.h"
#include "functional/nn_mv.h"
#include "functional/nn_neg.h"
#include "functional/nn_relu.h"
#include "functional/nn_relu6.h"
#include "functional/nn_rms_norm.h"
#include "functional/nn_softmax.h"
#include "functional/nn_silu.h"
#include "functional/nn_sub.h"
#include "functional/nn_sum.h"


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