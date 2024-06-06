

set(INCLUDES
  inc
)

set(SOURCES
  src/nn_tensor.c
  src/nn_print.c
  
  src/add/nn_add.c
  src/add/nn_add_rvv.c

  src/copy/nn_copy.c
  
  src/matmul/nn_matmul.c
  src/matmul/nn_matmul_eaglex.c
  src/matmul/nn_matmul_rvv.c
  
  src/max/nn_max.c
  src/max/nn_max_rvv.c
  
  src/min/nn_min.c
  src/min/nn_min_rvv.c

)

add_library(nn ${SOURCES})
target_include_directories(nn PUBLIC ${INCLUDES})

