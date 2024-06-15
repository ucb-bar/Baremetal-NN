
option(AVX "Use AVX implementation" OFF)
option(RVV "Use RISCV vector extension implementation" OFF)


add_library(nn 
  src/nn_tensor.c
  src/nn_print.c
  src/nn_abs.c
  src/nn_add.c
  src/batchnorm2d/nn_batchnorm2d.c
  src/conv2d/nn_conv2d.c
  src/nn_clip.c
  src/nn_copy.c
  src/nn_div.c
  src/nn_fill.c
  src/interpolate/nn_interpolate.c
  src/nn_linear.c
  src/nn_matmul.c
  src/matrixnorm/nn_matrixnorm.c
  src/nn_max.c
  src/nn_maximum.c
  src/maxpool2d/nn_maxpool2d.c
  src/nn_min.c
  src/nn_minimum.c
  src/nn_mul.c
  src/nn_neg.c
  src/nn_relu.c
  src/nn_relu6.c
  src/nn_sub.c
  src/nn_sum.c
  src/nn_transpose.c
)
target_include_directories(nn PUBLIC inc)


if (AVX)
  message(STATUS "Using AVX implementation")
  add_compile_definitions(AVX)
endif ()

if (RVV)
  message(STATUS "Using RVV implementation")
  add_compile_definitions(RVV)
endif ()

if (GEMMINI)
  message(STATUS "Using Gemmini implementation")
  add_compile_definitions(GEMMINI)
endif ()

if (X86)
  message("NN: Building for x86")
  target_link_libraries(nn target-x86)
    
elseif (RISCV)
  message("NN: Building for RISC-V")
  target_link_libraries(nn target-riscv)
endif ()


target_link_libraries(nn m)



