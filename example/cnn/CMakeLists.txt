
# the CMake files are still very hacky
# TODO: organize the build system properly

cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME        "cnn")

set(PROJECT_INCLUDES
    inc
)

set(PROJECT_SOURCES
    main.c
)

project(${PROJECT_NAME})


set(RISCV_DIR $ENV{RISCV})
set(CMAKE_CXX_COMPILER_FORCED TRUE)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Add optimization flags
set(CMAKE_BUILD_TYPE Debug)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O2")

set(STATIC_LINKING TRUE)

set(WRAP_SPECS_FILE "${RISCV_DIR}/riscv64-unknown-elf/lib/htif_wrap.specs")
set(SPECS_FILE "${RISCV_DIR}/riscv64-unknown-elf/lib/htif_nano.specs")
set(LIBGLOSS_DIR "${RISCV_DIR}/riscv64-unknown-elf/lib/")


# CMake toolchain definition for RISC-V GCC toolchain
set(CMAKE_SYSTEM_NAME "Generic" CACHE STRING "")
set(CMAKE_SYSTEM_PROCESSOR "riscv" CACHE STRING "")

set(TOOLCHAIN_PREFIX   "riscv64-unknown-elf-")

set(CMAKE_C_COMPILER   "${TOOLCHAIN_PREFIX}gcc")
set(CMAKE_ASM_COMPILER "${TOOLCHAIN_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PREFIX}g++")
set(CMAKE_AR           "${TOOLCHAIN_PREFIX}ar")
set(CMAKE_LINKER       "{TOOLCHAIN_PREFIX}ld")
set(CMAKE_OBJCOPY      "${TOOLCHAIN_PREFIX}objcopy")
set(CMAKE_SIZE         "${TOOLCHAIN_PREFIX}size")
set(CMAKE_STRIP        "${TOOLCHAIN_PREFIX}ld")


if(AVX)
    set(CMAKE_C_FLAGS "-march=native")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mcmodel=medany -march=rv64gcv_zfh -mabi=lp64d -fno-common -fno-builtin-printf")
set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS "-O2 -static -lm -lstdc++ -Wl,-Map=output.map -L${LIBGLOSS_DIR} -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE} -T ${CMAKE_SOURCE_DIR}/htif.ld")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -u _printf_float")

message(CMAKE_C_FLAGS: ${CMAKE_C_FLAGS})

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_INCLUDES})

add_subdirectory(../../nn ./build/nn)
target_link_libraries(${PROJECT_NAME} PUBLIC nn)

target_compile_options(${PROJECT_NAME} PRIVATE -u _printf_float)


