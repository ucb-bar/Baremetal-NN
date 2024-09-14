########################################################################################################################
# file:  CMakeLists.txt
#
# usage: 
#   Edit "VARIABLES"-section to suit project requirements.
#   Build instructions:
#     cmake -S ./ -B ./build/ -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -D RVV=ON
#     cmake --build ./build/ --target all
#     cmake --build ./build/ --target clean
#     cmake --install ./build/
########################################################################################################################
cmake_minimum_required(VERSION 3.10)

project(nn LANGUAGES C ASM)

# Options
option(BUILD_SHARED_LIBS    "Build using shared libraries"      OFF     )
option(CMAKE_BUILD_TYPE     "Build type"                        Debug   )
option(STATIC_LINKING       "Link statically"                   TRUE    )

# architecture options
option(X86                  "Build for x86"                     OFF     )
option(RISCV                "Build for RISC-V"                  OFF     )

# accelerator options
option(AVX                  "Use AVX implementation"            OFF     )
option(RVV                  "Use RISCV vector extension"        OFF     )
option(ZVFH                 "Use RISCV half-precision floating-point vector extension" OFF)

option(RVV_ASM              "Use RISCV vector extension in assembly implementation" OFF)

add_compile_options(-O1 -Wall -Wextra)

add_library(target-x86 INTERFACE)
target_compile_features(target-x86 INTERFACE c_std_11)
target_compile_definitions(target-x86 INTERFACE X86)

target_compile_options(target-x86 INTERFACE -march=native)
# target_compile_options(target-x86 INTERFACE -mavx512fp16)
target_link_options(target-x86 INTERFACE -static)
target_link_options(target-x86 INTERFACE -march=native)
target_link_options(target-x86 INTERFACE -fno-common -fno-builtin-printf)


add_library(target-riscv INTERFACE)
target_compile_features(target-riscv INTERFACE c_std_11)
target_compile_definitions(target-riscv INTERFACE RISCV)

set(WRAP_SPECS_FILE     "htif_wrap.specs")
set(SPECS_FILE          "htif_nano.specs")
# set(SPEC_FLAGS          -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE})

set(MARCH               "rv64gc")
set(MABI                "lp64d")
set(MCMODEL             "medany")

# generate march flags
if (RVV)
  list(APPEND MARCH "v")
  # list(APPEND MARCH "_zicntr")
  
  if (ZVFH)
    list(APPEND MARCH "_zfh")
    list(APPEND MARCH "_zvfh")
  endif()
endif()
if (RVV_ASM)
  list(APPEND MARCH "v")
endif()

list(JOIN MARCH "" MARCH)

if (NOT DEFINED LINKER_SCRIPT)
  set(LINKER_SCRIPT     ${CMAKE_SOURCE_DIR}/toolchain/htif.ld)
  # set(LINKER_SCRIPT     ${CMAKE_SOURCE_DIR}/toolchain/k230.ld)
endif()

target_compile_options(target-riscv INTERFACE -fno-common -fno-builtin-printf)
target_compile_options(target-riscv INTERFACE -march=${MARCH} -mabi=${MABI} -mcmodel=${MCMODEL})
target_compile_options(target-riscv INTERFACE -Wl,-Map=output.map ${SPEC_FLAGS})
target_compile_options(target-riscv INTERFACE -T ${LINKER_SCRIPT})

# target_compile_definitions(target-riscv INTERFACE FLT16_MAX=65504.0f)

target_link_options(target-riscv INTERFACE -static)
target_link_options(target-riscv INTERFACE -march=${MARCH} -mabi=${MABI} -mcmodel=${MCMODEL})
target_link_options(target-riscv INTERFACE -Wl,-Map=output.map ${SPEC_FLAGS})
target_link_options(target-riscv INTERFACE -T ${LINKER_SCRIPT})


set(cpu_impl 
  ./src/ops/cpu/abs.c
  ./src/ops/cpu/acc.c
  ./src/ops/cpu/acc1.c
  ./src/ops/cpu/add.c
  ./src/ops/cpu/add1.c
  ./src/ops/cpu/addmm.c
  ./src/ops/cpu/div.c
  ./src/ops/cpu/dot.c
  ./src/ops/cpu/fill.c
  ./src/ops/cpu/max.c
  ./src/ops/cpu/maximum.c
  ./src/ops/cpu/maximum1.c
  ./src/ops/cpu/mm.c
  ./src/ops/cpu/min.c
  ./src/ops/cpu/minimum.c
  ./src/ops/cpu/minimum1.c
  ./src/ops/cpu/mul.c
  ./src/ops/cpu/mul1.c
  ./src/ops/cpu/neg.c
  ./src/ops/cpu/norm.c
  ./src/ops/cpu/rms_norm.c
  ./src/ops/cpu/sgn.c
  ./src/ops/cpu/softmax.c
  ./src/ops/cpu/sqr.c
  ./src/ops/cpu/sqrt.c
  ./src/ops/cpu/sub.c
  ./src/ops/cpu/sum.c
  ./src/ops/cpu/transpose.c
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
    ./src/ops/rvv/abs.c
    ./src/ops/rvv/acc.c
    ./src/ops/rvv/acc1.c
    ./src/ops/rvv/add.c
    ./src/ops/rvv/add1.c
    ./src/ops/rvv/addmm.c
    ./src/ops/rvv/div.c
    ./src/ops/rvv/dot.c
    ./src/ops/rvv/max.c
    ./src/ops/rvv/maximum.c
    ./src/ops/rvv/maximum1.c
    ./src/ops/rvv/min.c
    ./src/ops/rvv/minimum.c
    ./src/ops/rvv/minimum1.c
    ./src/ops/rvv/mul.c
    ./src/ops/rvv/mul1.c
    ./src/ops/rvv/neg.c
    ./src/ops/rvv/rms_norm.c
    ./src/ops/rvv/sub.c
    ./src/ops/rvv/transpose.c
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
  ./src/functional/tensor_creation.c
  ./src/functional/print.c
  ./src/functional/abs.c
  ./src/functional/add.c
  ./src/functional/batch_norm2d.c
  ./src/functional/conv2d.c
  ./src/functional/clip.c
  ./src/functional/copy.c
  ./src/functional/div.c
  ./src/functional/elu.c
  ./src/functional/fill.c
  ./src/functional/interpolate.c
  ./src/functional/layer_norm.c
  ./src/functional/linear.c
  ./src/functional/matmul.c
  ./src/functional/mm.c
  ./src/functional/norm.c
  ./src/functional/max.c
  ./src/functional/maximum.c
  ./src/functional/max_pool2d.c
  ./src/functional/min.c
  ./src/functional/minimum.c
  ./src/functional/mul.c
  ./src/functional/mv.c
  ./src/functional/neg.c
  ./src/functional/relu.c
  ./src/functional/relu6.c
  ./src/functional/rms_norm.c
  ./src/functional/softmax.c
  ./src/functional/silu.c
  ./src/functional/sub.c
  ./src/functional/sum.c
  ./src/functional/transpose.c

  ${rvv_impl}
  ${gemmini_impl}
  ${cpu_impl}
)

target_include_directories(nn PUBLIC 
  ./nn
  ./nn/functional
  ./nn/ops)

if (X86)
  message(STATUS "nn: Building for x86")
  target_link_libraries(nn target-x86)
    
elseif (RISCV)
  message(STATUS "nn: Building for RISC-V")
  target_link_libraries(nn target-riscv)
endif ()


target_link_libraries(nn m)




add_subdirectory(tests)
add_subdirectory(examples)

SET_PROPERTY(TARGET nn PROPERTY PUBLIC_HEADER include/nn.h)

set(CMAKE_INSTALL_PREFIX "$ENV{RISCV}/riscv64-unknown-elf")

install(TARGETS nn 
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/nn
)
