#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"

#include "model.h"


Model model;


int main() {
  init(&model);

  ((float *)model.x.data)[0] = 0.0;

  forward(&model);

  printf("Output:\n");
  nn_printf(&model.fc3);

  return 0;
}