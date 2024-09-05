#ifndef __NN_H
#define __NN_H

#include <assert.h>

#include "float16.h"
#include "tensor.h"
#include "functional/tensor_creation.h"
#include "functional/print.h"
#include "functional/abs.h"
#include "functional/add.h"
#include "functional/batch_norm2d.h"
#include "functional/clip.h"
#include "functional/conv2d.h"
#include "functional/copy.h"
#include "functional/div.h"
#include "functional/elu.h"
#include "functional/fill.h"
#include "functional/interpolate.h"
#include "functional/layer_norm.h"
#include "functional/linear.h"
#include "functional/matmul.h"
#include "functional/norm.h"
#include "functional/max.h"
#include "functional/mm.h"
#include "functional/maximum.h"
#include "functional/min.h"
#include "functional/minimum.h"
#include "functional/mul.h"
#include "functional/mv.h"
#include "functional/neg.h"
#include "functional/relu.h"
#include "functional/relu6.h"
#include "functional/rms_norm.h"
#include "functional/softmax.h"
#include "functional/silu.h"
#include "functional/sub.h"
#include "functional/sum.h"


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