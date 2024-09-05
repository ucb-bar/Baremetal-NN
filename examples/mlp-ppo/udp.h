#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>


#define MICROSECOND_PER_SECOND       1000000


typedef struct {
  int sockfd;
  struct sockaddr_in recv_addr;
  struct sockaddr_in send_addr;
} UDP;


typedef struct {
  UDP udp;
  size_t n_obs;
  size_t n_acs;
} PolicyComm;

typedef struct {
  UDP udp;
  size_t n_obs;
  size_t n_acs;

  pthread_t thread_obs;
  pthread_t thread_acs;

  size_t frequency;

  float *obs;
  float *acs;
} EnvironmentComm;


ssize_t initialize_udp(
    UDP *udp,
    const char *recv_ip, const uint16_t recv_port,
    const char *send_ip, const uint16_t send_port
) {
  memset(udp, 0, sizeof(UDP));
  memset(&udp->recv_addr, 0, sizeof(udp->recv_addr));
  memset(&udp->send_addr, 0, sizeof(udp->send_addr));

  udp->recv_addr.sin_family = AF_INET;
  udp->recv_addr.sin_addr.s_addr = inet_addr(recv_ip);
  udp->recv_addr.sin_port = htons(recv_port);
  
  udp->send_addr.sin_family = AF_INET;
  udp->send_addr.sin_addr.s_addr = inet_addr(send_ip);
  udp->send_addr.sin_port = htons(send_port);

  if ((udp->sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    printf("[Error] <UDP> Error creating socket: %s\n", strerror(errno));
    return -1;
  }

  if (bind(udp->sockfd, (struct sockaddr *)&udp->recv_addr, sizeof(udp->recv_addr)) < 0) {
    printf("[Error] <UDP> Error binding socket: %s\n", strerror(errno));
    return -1;
  }

  printf("[INFO] <UDP> Server listening on %s:%d\n", recv_ip, recv_port);
  return 0;
}


ssize_t initialize_policy(
    PolicyComm *comm,
    const char *compute_ip, int compute_port,
    const char *env_ip, int env_port,
    size_t n_obs, size_t n_acs
) {
  memset(comm, 0, sizeof(PolicyComm));

  comm->n_obs = n_obs;
  comm->n_acs = n_acs;
  
  if (initialize_udp(&comm->udp, compute_ip, compute_port, env_ip, env_port) < 0) {
    printf("[Error] <UDP> Error initializing Policy communication\n");
    return 1;
  }

  return 0;
}

ssize_t receive_obs(PolicyComm *comm, float *obs) {
  size_t expected_bytes = sizeof(float) * comm->n_obs;
  ssize_t actual_bytes = recvfrom(comm->udp.sockfd, obs, expected_bytes, MSG_WAITALL, NULL, NULL);
  if (actual_bytes < 0 || actual_bytes != expected_bytes) {
    printf("[Error] <UDP> Error receiving: %s\n", strerror(errno));
  }
  return actual_bytes;
}

ssize_t send_acs(PolicyComm *comm, float *acs) {
  size_t expected_bytes = sizeof(float) * comm->n_acs;
  ssize_t actual_bytes = sendto(comm->udp.sockfd, acs, expected_bytes, 0, (const struct sockaddr *)&comm->udp.send_addr, sizeof(comm->udp.send_addr));
  if (actual_bytes < 0 || actual_bytes != expected_bytes) {
    printf("[Error] <UDP> Error sending: %s\n", strerror(errno));
  }
  return actual_bytes;
}




void action_handler(EnvironmentComm *comm) {
  size_t expected_bytes = sizeof(float) * comm->n_acs;

  while (1) {
    ssize_t actual_bytes = recvfrom(comm->udp.sockfd, comm->acs, expected_bytes, MSG_WAITALL, NULL, NULL);
    
    if (actual_bytes < 0 || actual_bytes != expected_bytes) {
      printf("[Error] <UDP> Error receiving action: %s\n", strerror(errno));
      continue;
    }
  }
}

void observation_handler(EnvironmentComm *comm) {
  struct timeval start_time, current_time;
  long elapsed_us;

  size_t expected_bytes = sizeof(float) * comm->n_obs;

  while (1) {
    gettimeofday(&start_time, NULL);

    ssize_t actual_bytes = sendto(comm->udp.sockfd, comm->obs, expected_bytes, 0, (const struct sockaddr *)&comm->udp.send_addr, sizeof(comm->udp.send_addr));
    if (actual_bytes < 0 || actual_bytes != expected_bytes) {
      printf("[Error] <UDP> Error sending obs: %s\n", strerror(errno));
      // sleep for 100ms before retrying
      usleep(100000);
    }

    gettimeofday(&current_time, NULL);
    elapsed_us = (current_time.tv_sec - start_time.tv_sec) * MICROSECOND_PER_SECOND
               + (current_time.tv_usec - start_time.tv_usec);

    // Sleep for the remaining time to achieve target Hz rate
    long elapsed_us = MICROSECOND_PER_SECOND / comm->frequency;
    if (elapsed_us < elapsed_us) {
      struct timespec ts;
      ts.tv_sec = 0;
      ts.tv_nsec = elapsed_us - elapsed_us;
      nanosleep(&ts, NULL);
    }
  }
}




ssize_t initialize_env(
    EnvironmentComm *comm,
    const char *compute_ip, int compute_port,
    const char *env_ip, int env_port,
    size_t n_obs, size_t n_acs,
    float *obs, float *acs,
    size_t frequency
) {
  memset(comm, 0, sizeof(EnvironmentComm));
  
  comm->obs = obs;
  comm->acs = acs;
  comm->n_obs = n_obs;
  comm->n_acs = n_acs;
  comm->frequency = frequency;

  if (initialize_udp(&comm->udp, env_ip, env_port, compute_ip, compute_port) < 0) {
    printf("[Error] <UDP> Error initializing Environment communication\n");
    return 1;
  }
  
  if (pthread_create(&comm->thread_obs, NULL, (void *)observation_handler, (void *)comm) != 0) {
    printf("[Error] <UDP> Error creating observation_handler thread\n");
    return 1;
  }

  if (pthread_create(&comm->thread_acs, NULL, (void *)action_handler, (void *)comm) != 0) {
    printf("[Error] <UDP> Error creating action_handler thread\n");
    return 1;
  }

  return 0;
}

void receive_acs(EnvironmentComm *comm, float *acs) {
  memcpy(acs, comm->acs, sizeof(float) * comm->n_acs);
}

void send_obs(EnvironmentComm *comm, float *obs) {
  memcpy(comm->obs, obs, sizeof(float) * comm->n_obs);
}
