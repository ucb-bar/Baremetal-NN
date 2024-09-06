########################################################################################################################
# file:  CMakeLists.txt
#
# usage: 
#   Edit "VARIABLES"-section to suit project requirements.
#   Build instructions:
#     cmake . -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -S ./ -B ./build/ -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug -D RVV=ON
#     cmake --build ./build/ --target all
#     cmake --build ./build/ --target clean
########################################################################################################################
cmake_minimum_required(VERSION 3.10)

project(nn LANGUAGES C)

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

set(MARCH               "rv64gc")
set(MABI                "lp64d")
set(MCMODEL             "medany")

# generate march flags
if (RVV)
  list(APPEND MARCH "v")
  list(APPEND MARCH "_zicntr")
  
  if (ZVFH)
    list(APPEND MARCH "_zfh")
    list(APPEND MARCH "_zvfh")
  endif()
endif()

list(JOIN MARCH "" MARCH)

if (NOT DEFINED LINKER_SCRIPT)
  set(LINKER_SCRIPT     ${CMAKE_SOURCE_DIR}/toolchain/htif.ld)
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


add_subdirectory(nn)
add_subdirectory(tests)
add_subdirectory(examples)

SET_PROPERTY(TARGET XNNPACK PROPERTY PUBLIC_HEADER include/nn.h)

set(CMAKE_INSTALL_PREFIX "$ENV{RISCV}/riscv64-unknown-elf")

install(TARGETS nn 
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/nn
)
