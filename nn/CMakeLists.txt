

set(cpu_impl 
  ./impl/cpu/abs.c
  ./impl/cpu/acc.c
  ./impl/cpu/acc1.c
  ./impl/cpu/add.c
  ./impl/cpu/add1.c
  ./impl/cpu/div.c
  ./impl/cpu/dot.c
  ./impl/cpu/fill.c
  ./impl/cpu/max.c
  ./impl/cpu/maximum.c
  ./impl/cpu/maximum1.c
  ./impl/cpu/min.c
  ./impl/cpu/minimum.c
  ./impl/cpu/minimum1.c
  ./impl/cpu/mul.c
  ./impl/cpu/mul1.c
  ./impl/cpu/neg.c
  ./impl/cpu/norm.c
  ./impl/cpu/rms_norm.c
  ./impl/cpu/sgn.c
  ./impl/cpu/softmax.c
  ./impl/cpu/sqr.c
  ./impl/cpu/sqrt.c
  ./impl/cpu/sub.c
  ./impl/cpu/sum.c
  ./impl/cpu/transpose.c
)


if (AVX)
  message(STATUS "Using AVX implementation")
  add_compile_definitions(AVX)
endif ()

if (RVV)
  message(STATUS "Using RVV implementation")
  add_compile_definitions(RVV)
  
  set(rvv_impl
    ./impl/rvv/abs.c
    ./impl/rvv/acc.c
    ./impl/rvv/acc1.c
    ./impl/rvv/add.c
    ./impl/rvv/add1.c
    ./impl/rvv/div.c
    ./impl/rvv/dot.c
    ./impl/rvv/max.c
    ./impl/rvv/maximum.c
    ./impl/rvv/maximum1.c
    ./impl/rvv/min.c
    ./impl/rvv/minimum.c
    ./impl/rvv/minimum1.c
    ./impl/rvv/mul.c
    ./impl/rvv/mul1.c
    ./impl/rvv/neg.c
    ./impl/rvv/rms_norm.c
    ./impl/rvv/sub.c
    ./impl/rvv/transpose.c
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
  ./functional/nn_tensor_creation.c
  ./functional/nn_print.c
  ./functional/nn_abs.c
  ./functional/nn_add.c
  ./functional/nn_batch_norm2d.c
  ./functional/nn_conv2d.c
  ./functional/nn_clip.c
  ./functional/nn_copy.c
  ./functional/nn_div.c
  ./functional/nn_elu.c
  ./functional/nn_fill.c
  ./functional/nn_interpolate.c
  ./functional/nn_layer_norm.c
  ./functional/nn_linear.c
  ./functional/nn_matmul.c
  ./functional/nn_mm.c
  ./functional/nn_norm.c
  ./functional/nn_max.c
  ./functional/nn_maximum.c
  ./functional/nn_max_pool2d.c
  ./functional/nn_min.c
  ./functional/nn_minimum.c
  ./functional/nn_mul.c
  ./functional/nn_mv.c
  ./functional/nn_neg.c
  ./functional/nn_relu.c
  ./functional/nn_relu6.c
  ./functional/nn_rms_norm.c
  ./functional/nn_softmax.c
  ./functional/nn_silu.c
  ./functional/nn_sub.c
  ./functional/nn_sum.c
  ./functional/nn_transpose.c

  ${rvv_impl}
  ${gemmini_impl}
  ${cpu_impl}
)

target_include_directories(nn PUBLIC 
  ./
  ./functional 
  ./impl)

if (X86)
  message(STATUS "nn: Building for x86")
  target_link_libraries(nn target-x86)
    
elseif (RISCV)
  message(STATUS "nn: Building for RISC-V")
  target_link_libraries(nn target-riscv)
endif ()


target_link_libraries(nn m)



