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
option(RISCV_V              "Use RISCV vector extension"        OFF     )
option(RISCV_ZVFH           "Use RISCV half-precision floating-point vector extension" OFF)
option(RISCV_V_ASM          "Use RISCV vector extension in assembly implementation" OFF)

option(DEBUG_USE_REDOSUM        "Use redosum instead of redusum" OFF)


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
set(SPEC_FLAGS          -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE})

set(ARCH               "rv64gc")
set(ABI                "lp64d")
set(CMODEL             "medany")

# generate march flags
if (RISCV_V)
  list(APPEND ARCH "v")
  # list(APPEND MARCH "_zicntr")
  
  if (RISCV_ZVFH)
    list(APPEND ARCH "_zfh")
    list(APPEND ARCH "_zvfh")
  endif()
endif()

list(JOIN ARCH "" ARCH)

if (NOT DEFINED LINKER_SCRIPT)
  set(LINKER_SCRIPT     ${CMAKE_SOURCE_DIR}/toolchain/htif.ld)
  # set(LINKER_SCRIPT     ${CMAKE_SOURCE_DIR}/toolchain/k230.ld)
endif()

target_compile_options(target-riscv INTERFACE -fno-common -fno-builtin-printf)
target_compile_options(target-riscv INTERFACE -march=${ARCH} -mabi=${ABI} -mcmodel=${CMODEL})
target_compile_options(target-riscv INTERFACE -Wl,-Map=output.map ${SPEC_FLAGS})
target_compile_options(target-riscv INTERFACE -T ${LINKER_SCRIPT})

# target_compile_definitions(target-riscv INTERFACE FLT16_MAX=65504.0f)

target_link_options(target-riscv INTERFACE -static)
target_link_options(target-riscv INTERFACE -march=${ARCH} -mabi=${ABI} -mcmodel=${CMODEL})
target_link_options(target-riscv INTERFACE -Wl,-Map=output.map ${SPEC_FLAGS})
target_link_options(target-riscv INTERFACE -T ${LINKER_SCRIPT})


set(cpu_impl 
  ./src/print.c
  ./src/cpu/add.c
  ./src/cpu/addmm.c
  ./src/cpu/addscalar.c
  ./src/cpu/dot.c
  ./src/cpu/elu.c
  ./src/cpu/equals.c
  ./src/cpu/max.c
  ./src/cpu/min.c
  ./src/cpu/mm.c
  ./src/cpu/mul.c
  ./src/cpu/mulscalar.c
  ./src/cpu/relu.c
  ./src/cpu/tanh.c
)


if (AVX)
  message(STATUS "Using AVX implementation")
  add_compile_definitions(AVX)
endif ()

if (RISCV_V)
  message(STATUS "Using RVV implementation")
  add_compile_definitions(RISCV_V)
  
  if (RISCV_ZVFH)
    message(STATUS "Using Zvfh extension")
    add_compile_definitions(RISCV_ZVFH)
  endif ()
  
  if (RISCV_V_ASM)
    message(STATUS "Using RISCV_V assembly implementation")
    add_compile_definitions(RISCV_V_ASM)

    set(rvv_asm_impl
      ./src/rvv/add.S
      ./src/rvv/addmm.S
      ./src/rvv/relu.S
      ./src/rvv/mm.S
    )
  endif ()

  if (DEBUG_USE_REDOSUM)
    message(STATUS "Using redosum implementation")
    add_compile_definitions(DEBUG_USE_REDOSUM)
  endif ()

  set(rvv_impl
    ./src/rvv/add.c
    ./src/rvv/addscalar.c
    ./src/rvv/addmm.c
    ./src/rvv/elu.c
    ./src/rvv/relu.c
    ./src/rvv/max.c
    ./src/rvv/min.c
    ./src/rvv/mm.c
    ./src/rvv/mul.c
    ./src/rvv/mulscalar.c

    ${rvv_asm_impl}
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
  ${rvv_impl}
  ${gemmini_impl}
  ${cpu_impl}
)

target_include_directories(nn PUBLIC 
  ./nn
)
  # ./nn/functional
  # ./nn/ops)

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
