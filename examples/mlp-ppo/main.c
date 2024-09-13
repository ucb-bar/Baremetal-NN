/**
 * @file main.c
 * 
 * Running MLP neural network robot control policy on the target compute device
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rv.h"
#include "nn.h"
#include "model.h"

#include "udp.h"



#define N_OBSERVATIONS    123
#define N_ACTIONS         37

#define ENV_IP            "172.28.0.2"
#define ENV_PORT          8010
#define POLICY_IP         "0.0.0.0"
#define POLICY_PORT       8011


int main() {

  Model *model = malloc(sizeof(Model));

  // get the input and output layer tensor data pointer
  float obs[N_OBSERVATIONS];
  float acs[N_ACTIONS];

  size_t counter = 0;
  
  printf("Initalizing model...\n");
  init(model);


  printf("Initalizing UDP communication...\n");
  PolicyComm comm;
  initialize_policy(
    &comm,
    POLICY_IP, POLICY_PORT,
    ENV_IP, ENV_PORT,
    N_OBSERVATIONS, N_ACTIONS
  );


  // for performance measurement
  struct timespec inference_start_time;
  struct timespec inference_end_time;


  printf("Policy started. Waiting for environment...\n");

  while (1) {
    /* receive */
    receive_obs(&comm, obs);
    

    /* forward */
    clock_gettime(CLOCK_REALTIME, &inference_start_time);

    memcpy(model->input_1.data, obs, N_OBSERVATIONS * sizeof(float));
    forward(model);
    memcpy(acs, model->_6.data, N_ACTIONS * sizeof(float));
    
    clock_gettime(CLOCK_REALTIME, &inference_end_time);
    

    /* transmit */
    send_acs(&comm, acs);
    

    /* log */
    
    // calculate elapsed time
    struct timespec elapsed_time;

    // handle nanosecond borrowing
    if ((inference_end_time.tv_nsec - inference_start_time.tv_nsec) < 0) {
      elapsed_time.tv_sec = inference_end_time.tv_sec - inference_start_time.tv_sec - 1;
      elapsed_time.tv_nsec = 1000000000 + inference_end_time.tv_nsec - inference_start_time.tv_nsec;
    } else {
      elapsed_time.tv_sec = inference_end_time.tv_sec - inference_start_time.tv_sec;
      elapsed_time.tv_nsec = inference_end_time.tv_nsec - inference_start_time.tv_nsec;
    }

    float seconds = elapsed_time.tv_sec + elapsed_time.tv_nsec / 1e9;

    // clear lines
    for (size_t i=0; i<10; i+=1) {
      printf("\033[F\033[K");
    }

    printf("+------------------+---------------------------------------------------------------------------+\n");
    printf("| Loop             | %8d                                                                  |\n", counter);
    printf("|==================+===========================================================================|\n");
    printf("| Observations     | ");
    for (size_t i=0; i<10; i+=1) {
      printf("%6.3f ", obs[i]);
    }
    printf("    |\n");
    printf("|------------------+---------------------------------------------------------------------------|\n");
    printf("| Actions          | ");
    for (size_t i=0; i<10; i+=1) {
      printf("%6.3f ", acs[i]);
    }
    printf("    |\n");
    printf("|==================+===========================================================================|\n");
    printf("| Inference speed  | %.3f Hz  (%6.3f ms)                                                   |\n", 1.f / seconds, seconds * 1000.f);
    printf("+------------------+---------------------------------------------------------------------------+\n");
    printf("\n");

    counter += 1;
  }
}
