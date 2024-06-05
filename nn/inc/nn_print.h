#ifndef __NN_PRINT_H
#define __NN_PRINT_H

#include "nn_tensor.h"


void NN_printFloat(float v, int16_t num_digits);

void NN_printShape(Tensor *t);

void NN_printf(Tensor *t);


#endif // __NN_PRINT_H
