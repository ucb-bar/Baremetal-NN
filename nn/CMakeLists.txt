cmake_minimum_required(VERSION 3.10)

set(cpu_impl 
  ./kernel/cpu/abs.c
  ./kernel/cpu/acc.c
  ./kernel/cpu/acc1.c
  ./kernel/cpu/add.c
  ./kernel/cpu/add1.c
  ./kernel/cpu/div.c
  ./kernel/cpu/dot.c
  ./kernel/cpu/fill.c
  ./kernel/cpu/max.c
  ./kernel/cpu/maximum.c
  ./kernel/cpu/maximum1.c
  ./kernel/cpu/min.c
  ./kernel/cpu/minimum.c
  ./kernel/cpu/minimum1.c
  ./kernel/cpu/mul.c
  ./kernel/cpu/mul1.c
  ./kernel/cpu/neg.c
  ./kernel/cpu/norm.c
  ./kernel/cpu/rms_norm.c
  ./kernel/cpu/sgn.c
  ./kernel/cpu/softmax.c
  ./kernel/cpu/sqr.c
  ./kernel/cpu/sqrt.c
  ./kernel/cpu/sub.c
  ./kernel/cpu/sum.c
  ./kernel/cpu/transpose.c
)


if (AVX)
  message(STATUS "Using AVX implementation")
  add_compile_definitions(AVX)
endif ()

if (RVV)
  message(STATUS "Using RVV implementation")
  add_compile_definitions(RVV)
  
  if (RISCV_ZVFH)
    message(STATUS "Using Zvfh extension")
    add_compile_definitions(RISCV_ZVFH)
  endif ()
  
  set(rvv_impl
    ./kernel/rvv/abs.c
    ./kernel/rvv/acc.c
    ./kernel/rvv/acc1.c
    ./kernel/rvv/add.c
    ./kernel/rvv/add1.c
    ./kernel/rvv/div.c
    ./kernel/rvv/dot.c
    ./kernel/rvv/max.c
    ./kernel/rvv/maximum.c
    ./kernel/rvv/maximum1.c
    ./kernel/rvv/min.c
    ./kernel/rvv/minimum.c
    ./kernel/rvv/minimum1.c
    ./kernel/rvv/mul.c
    ./kernel/rvv/mul1.c
    ./kernel/rvv/neg.c
    ./kernel/rvv/rms_norm.c
    ./kernel/rvv/sub.c
    ./kernel/rvv/transpose.c
    )
endif ()

if (GEMMINI)
  message(STATUS "Using Gemmini implementation")
  add_compile_definitions(GEMMINI)

  set(gemmini_impl
      impl/gemmini/mm.c
      )
endif ()


add_library(nn
  ./layer/tensor_creation.c
  ./layer/print.c
  ./layer/abs.c
  ./layer/add.c
  ./layer/batch_norm2d.c
  ./layer/conv2d.c
  ./layer/clip.c
  ./layer/copy.c
  ./layer/div.c
  ./layer/elu.c
  ./layer/fill.c
  ./layer/interpolate.c
  ./layer/layer_norm.c
  ./layer/linear.c
  ./layer/matmul.c
  ./layer/mm.c
  ./layer/norm.c
  ./layer/max.c
  ./layer/maximum.c
  ./layer/max_pool2d.c
  ./layer/min.c
  ./layer/minimum.c
  ./layer/mul.c
  ./layer/mv.c
  ./layer/neg.c
  ./layer/relu.c
  ./layer/relu6.c
  ./layer/rms_norm.c
  ./layer/softmax.c
  ./layer/silu.c
  ./layer/sub.c
  ./layer/sum.c
  ./layer/transpose.c

  ${rvv_impl}
  ${gemmini_impl}
  ${cpu_impl}
)

target_include_directories(nn PUBLIC ./)

if (X86)
  message(STATUS "nn: Building for x86")
  target_link_libraries(nn target-x86)
    
elseif (RISCV)
  message(STATUS "nn: Building for RISC-V")
  target_link_libraries(nn target-riscv)
endif ()


target_link_libraries(nn m)
