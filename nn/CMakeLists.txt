
option(RVV "Use RISCV vector extension implementation" OFF)

set(INCLUDES
  inc
)

set(SOURCES
  src/nn_tensor.c
  src/nn_print.c
  src/add/nn_add.c
  src/conv2d/nn_conv2d.c
  src/copy/nn_copy.c
  src/linear/nn_linear.c
  src/matmul/nn_matmul.c
  src/max/nn_max.c
  src/maxpool2d/nn_maxpool2d.c
  src/min/nn_min.c
  src/mul/nn_mul.c
  src/relu/nn_relu.c
  src/sub/nn_sub.c
  src/sum/nn_sum.c
  src/transpose/nn_transpose.c
)

set(RVV_SOURCE
  src/add/nn_add_rvv.c
  src/linear/nn_linear_rvv.c
  src/matmul/nn_matmul_rvv.c
  src/max/nn_max_rvv.c
  src/min/nn_min_rvv.c
  src/mul/nn_mul_rvv.c
  src/sub/nn_sub_rvv.c
)

if(RVV)
  set(SOURCES ${SOURCES} ${RVV_SOURCE})
  message(STATUS "Using RVV implementation")
endif(RVV)

add_library(nn ${SOURCES})



target_include_directories(nn PUBLIC ${INCLUDES})

