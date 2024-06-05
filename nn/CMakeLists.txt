

set(INCLUDES
  inc
)

set(SOURCES
  src/nn_tensor.c
  src/nn_print.c
  src/add/nn_add.c
  src/copy/nn_copy.c
  src/matmul/nn_matmul.c
  
  src/add/nn_add_rvv.c
  src/matmul/nn_matmul_rvv.c
)

add_library(nn ${SOURCES})
target_include_directories(nn PUBLIC ${INCLUDES})

