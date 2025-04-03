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
option(CONFIG_TOOLCHAIN_RISCV           "Build for RISC-V"                    OFF     )
option(CONFIG_TOOLCHAIN_NATIVE          "Build for native"                    ON      )

# dtype options
option(CONFIG_DTYPE_ENABLE_F16          "Enable F16 support"                  ON      )
option(CONFIG_DTYPE_ENABLE_I32          "Enable I32 support"                  ON      )

# accelerator options
option(CONFIG_BACKEND_RISCV_V           "Use RISCV vector extension"          OFF     )
option(CONFIG_BACKEND_RISCV_ZVFH        "Use RISCV half-precision floating-point vector extension"  OFF )
option(RISCV_V_ASM          "Use RISCV vector extension in assembly implementation"                 OFF )
option(CONFIG_BACKEND_GEMMINI           "Use Gemmini implementation"          OFF     )

# debug options
# for RISC-V vector implementation
option(CONFIG_DEBUG_USE_REDOSUM        "Use redosum instead of redusum" OFF)


add_compile_options(-O1 -Wall -Wextra -Wno-unused-variable)

add_library(target-native INTERFACE)
target_compile_features(target-native INTERFACE c_std_11)
target_compile_options(target-native INTERFACE -march=native)
target_link_options(target-native INTERFACE -static)
target_link_options(target-native INTERFACE -march=native)
target_link_options(target-native INTERFACE -fno-common -fno-builtin-printf)


add_library(target-riscv INTERFACE)
target_compile_features(target-riscv INTERFACE c_std_11)

set(WRAP_SPECS_FILE     "htif_wrap.specs")
set(SPECS_FILE          "htif_nano.specs")
set(SPEC_FLAGS          -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE})

set(ARCH               "rv64gc")
set(ABI                "lp64d")
set(CMODEL             "medany")

# generate march flags
if (CONFIG_BACKEND_RISCV_V)
  list(APPEND ARCH "v")
  # list(APPEND MARCH "_zicntr")
  
  if (CONFIG_BACKEND_RISCV_ZVFH)
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

target_compile_definitions(target-riscv INTERFACE CONFIG_TOOLCHAIN_RISCV=1)
# target_compile_definitions(target-riscv INTERFACE FLT16_MAX=65504.0f)

target_link_options(target-riscv INTERFACE -static)
target_link_options(target-riscv INTERFACE -march=${ARCH} -mabi=${ABI} -mcmodel=${CMODEL})
target_link_options(target-riscv INTERFACE -Wl,-Map=output.map ${SPEC_FLAGS})
target_link_options(target-riscv INTERFACE -T ${LINKER_SCRIPT})


add_library(nn INTERFACE)


if (CONFIG_TOOLCHAIN_NATIVE)
  message(STATUS "nn: Building with native toolchain")
  target_link_libraries(nn INTERFACE target-native)
elseif (CONFIG_TOOLCHAIN_RISCV)
  message(STATUS "nn: Building with RISC-V toolchain")
  target_link_libraries(nn INTERFACE target-riscv)
endif ()


if (CONFIG_DTYPE_ENABLE_F16)
  target_compile_definitions(nn INTERFACE CONFIG_DTYPE_ENABLE_F16)
endif()

if (CONFIG_DTYPE_ENABLE_I32)
  target_compile_definitions(nn INTERFACE CONFIG_DTYPE_ENABLE_I32)
endif()


if (CONFIG_BACKEND_RISCV_V)
  message(STATUS "Using RVV implementation")
  add_compile_definitions(CONFIG_BACKEND_RISCV_V)
  
  if (CONFIG_BACKEND_RISCV_ZVFH)
    message(STATUS "Using Zvfh extension")
    add_compile_definitions(CONFIG_BACKEND_RISCV_ZVFH)
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

  if (CONFIG_DEBUG_USE_REDOSUM)
    message(STATUS "Using redosum implementation")
    add_compile_definitions(CONFIG_DEBUG_USE_REDOSUM)
  endif ()

endif ()

if (CONFIG_BACKEND_GEMMINI)
  message(STATUS "Using Gemmini implementation")
  add_compile_definitions(CONFIG_BACKEND_GEMMINI)
endif ()


target_include_directories(nn INTERFACE
  ./nn
  ./device
  ./device/gemmini
  )


target_link_libraries(nn INTERFACE m)


add_subdirectory(tests)
add_subdirectory(examples)


SET_PROPERTY(TARGET nn PROPERTY PUBLIC_HEADER
  nn/nn.h
  nn/nn_f32.h
  nn/nn_f16.h
  nn/nn_i32.h
)

set(CMAKE_INSTALL_PREFIX "$ENV{RISCV}/riscv64-unknown-elf")

install(TARGETS nn 
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/nn
)
