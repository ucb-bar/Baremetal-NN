
option(AVX "Use AVX implementation" OFF)
option(RVV "Use RISCV vector extension implementation" OFF)

set(cpu_impl 
  impl/cpu/abs.c
  impl/cpu/acc.c
  impl/cpu/acc1.c
  impl/cpu/add.c
  impl/cpu/add1.c
  impl/cpu/div.c
  impl/cpu/dot.c
  impl/cpu/fill.c
  impl/cpu/max.c
  impl/cpu/maximum.c
  impl/cpu/maximum1.c
  impl/cpu/min.c
  impl/cpu/minimum.c
  impl/cpu/minimum1.c
  impl/cpu/mul.c
  impl/cpu/mul1.c
  impl/cpu/neg.c
  impl/cpu/norm.c
  impl/cpu/rmsnorm.c
  impl/cpu/sgn.c
  impl/cpu/softmax.c
  impl/cpu/sqr.c
  impl/cpu/sqrt.c
  impl/cpu/sub.c
  impl/cpu/sum.c
  impl/cpu/transpose.c
)

set(rvv_impl
  impl/rvv/abs.c
  impl/rvv/acc.c
  impl/rvv/acc1.c
  impl/rvv/add.c
  impl/rvv/add1.c
  impl/rvv/div.c
  impl/rvv/max.c
  impl/rvv/maximum.c
  impl/rvv/maximum1.c
  impl/rvv/min.c
  impl/rvv/minimum.c
  impl/rvv/minimum1.c
  impl/rvv/mul.c
  impl/rvv/mul1.c
  impl/rvv/neg.c
  impl/rvv/sub.c
  impl/rvv/transpose.c
)


add_library(nn
  op/nn_tensor_creation.c
  op/nn_print.c
  op/nn_abs.c
  op/nn_add.c
  op/nn_batch_norm2d.c
  op/nn_conv2d.c
  op/nn_clip.c
  op/nn_copy.c
  op/nn_div.c
  op/nn_elu.c
  op/nn_fill.c
  op/nn_interpolate.c
  op/nn_layer_norm.c
  op/nn_linear.c
  op/nn_matmul.c
  op/nn_mm.c
  op/nn_norm.c
  op/nn_max.c
  op/nn_maximum.c
  op/nn_max_pool2d.c
  op/nn_min.c
  op/nn_minimum.c
  op/nn_mul.c
  op/nn_neg.c
  op/nn_relu.c
  op/nn_relu6.c
  op/nn_softmax.c
  op/nn_sub.c
  op/nn_sum.c
  op/nn_transpose.c
  op/nn_unfold.c

  ${cpu_impl}
  ${rvv_impl}
)

target_include_directories(nn PUBLIC op impl)


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



