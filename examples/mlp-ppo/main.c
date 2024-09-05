/**
 * @file main.c
 * 
 * A simple example demonstrating C = A * B + D
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rv.h"
#include "nn.h"
#include "model.h"

#include "udp.h"



#define ENV_IP            "10.0.0.1"
#define ENV_PORT          8010
#define POLICY_IP         "0.0.0.0"
#define POLICY_PORT       8011


int main() {

  Model *model = malloc(sizeof(Model));


  size_t cycles;
  
  printf("initalizing model...\n");
  init(model);


  PolicyComm comm;

  initialize_policy(
    &comm,
    POLICY_IP, POLICY_PORT,
    ENV_IP, ENV_PORT,
    123, 37
  );

  float obs[256];
  float acs[37];

  for (int i=0; i<37; i+=1) {
    acs[i] = 0;
  }

  printf("waiting for  obs...\n");
  while (1) {
  
    receive_obs(&comm, model->input_1.data);

    // cycles = READ_CSR("mcycle");
    forward(model);
    // cycles = READ_CSR("mcycle") - cycles;

    printf("cycles: %lu\n", cycles);

    // printf("output:\n");
    // NN_printf(&model->_6);

    send_acs(&comm, model->_6.data);
  }

  
}
