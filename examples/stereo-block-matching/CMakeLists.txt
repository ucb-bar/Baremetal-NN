cmake_minimum_required(VERSION 3.10)

project(stereoBM LANGUAGES C)


add_executable(stereoBM main.c)
target_include_directories(stereoBM PUBLIC inc ../utility)

target_compile_features(stereoBM INTERFACE c_std_11)

if (X86)
    message("stereoBM: building for x86")
    target_link_libraries(stereoBM PUBLIC target-x86)

elseif (RISCV)
    message("stereoBM: building for RISC-V")
    # CMake toolchain definition for RISC-V GCC toolchain
    set(CMAKE_SYSTEM_NAME "Generic" CACHE STRING "")
    set(CMAKE_SYSTEM_PROCESSOR "riscv" CACHE STRING "")

    set(TOOLCHAIN_PREFIX    "riscv64-unknown-elf-")

    set(CMAKE_C_COMPILER    "${TOOLCHAIN_PREFIX}gcc")
    set(CMAKE_ASM_COMPILER  "${TOOLCHAIN_PREFIX}gcc")
    set(CMAKE_CXX_COMPILER  "${TOOLCHAIN_PREFIX}g++")
    set(CMAKE_AR            "${TOOLCHAIN_PREFIX}ar")
    set(CMAKE_LINKER        "{TOOLCHAIN_PREFIX}ld")
    set(CMAKE_OBJCOPY       "${TOOLCHAIN_PREFIX}objcopy")
    set(CMAKE_SIZE          "${TOOLCHAIN_PREFIX}size")
    set(CMAKE_STRIP         "${TOOLCHAIN_PREFIX}ld")
    
    target_link_libraries(stereoBM PUBLIC target-riscv)
endif ()

add_compile_options(-O3 -Wall -Wextra)

target_compile_options(stereoBM PRIVATE -u _printf_float)

add_subdirectory(../../ ./build/)

add_subdirectory(../../nn/ ./build/nn)
target_link_libraries(stereoBM PUBLIC nn)

target_link_libraries(stereoBM PUBLIC m)

