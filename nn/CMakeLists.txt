
option(AVX "Use AVX implementation" OFF)
option(RVV "Use RISCV vector extension implementation" OFF)

set(SOURCES
  src/nn_tensor.c
  src/nn_print.c
  src/abs/nn_abs.c
  src/add/nn_add.c
  src/batchnorm2d/nn_batchnorm2d.c
  src/conv2d/nn_conv2d.c
  src/copy/nn_copy.c
  src/interpolate/nn_interpolate.c
  src/linear/nn_linear.c
  src/matmul/nn_matmul.c
  src/matrixnorm/nn_matrixnorm.c
  src/max/nn_max.c
  src/maximum/nn_maximum.c
  src/maxpool2d/nn_maxpool2d.c
  src/min/nn_min.c
  src/minimum/nn_minimum.c
  src/mul/nn_mul.c
  src/neg/nn_neg.c
  src/relu/nn_relu.c
  src/relu6/nn_relu6.c
  src/sub/nn_sub.c
  src/sum/nn_sum.c
  src/transpose/nn_transpose.c
)

set(AVX_SOURCE
  src/abs/nn_abs_avx.c
)

set(RVV_SOURCE
  src/abs/nn_abs_rvv.c
  src/add/nn_add_rvv.c
  src/conv2d/nn_conv2d_rvv.c
  src/linear/nn_linear_rvv.c
  src/matmul/nn_matmul_rvv.c
  src/matrixnorm/nn_matrixnorm_rvv.c
  src/max/nn_max_rvv.c
  src/maximum/nn_maximum_rvv.c
  src/min/nn_min_rvv.c
  src/minimum/nn_minimum_rvv.c
  src/mul/nn_mul_rvv.c
  src/neg/nn_neg_rvv.c
  src/sub/nn_sub_rvv.c
)

if(AVX)
  set(SOURCES ${SOURCES} ${AVX_SOURCE})
  message(STATUS "Using AVX implementation")
endif(AVX)

if(RVV)
  set(SOURCES ${SOURCES} ${RVV_SOURCE})
  message(STATUS "Using RVV implementation")
endif(RVV)

add_library(nn ${SOURCES})
target_include_directories(nn PUBLIC inc)

if (X86)
    message("NN: Building for x86")
    target_link_libraries(nn target-x86)
    
elseif (RISCV)
    message("NN: Building for RISC-V")
    target_link_libraries(nn target-riscv)
endif ()


target_link_libraries(nn m)



