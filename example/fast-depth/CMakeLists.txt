# the CMake files are still very hacky
# TODO: organize the build system properly

cmake_minimum_required(VERSION 3.10)

project(fast-depth LANGUAGES C)


# Options
option(BUILD_SHARED_LIBS    "Build using shared libraries"      OFF     )
option(CMAKE_BUILD_TYPE     "Build type"                        Debug   )
option(STATIC_LINKING       "Link statically"                   TRUE    )

option(X86                  "Build for x86"                     OFF     )
option(RISCV                "Build for RISC-V"                  OFF     )


add_executable(fast-depth main.c)
target_include_directories(fast-depth PUBLIC inc)

target_compile_features(fast-depth INTERFACE c_std_11)


set(CMAKE_C_FLAGS           "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG     "-O2")
set(CMAKE_C_FLAGS_RELEASE   "-O3")
set(CMAKE_EXE_LINKER_FLAGS  "-static")

if (X86)
    message("Building for x86")
    set(CMAKE_C_FLAGS       "${CMAKE_C_FLAGS} -march=native")
elseif (RISCV)
    message("Building for RISC-V")
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

    set(WRAP_SPECS_FILE     "htif_wrap.specs")
    set(SPECS_FILE          "htif_nano.specs")
    set(LIBGLOSS_DIR        "$ENV{RISCV}/riscv64-unknown-elf/lib/")

    set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS} -fno-common -fno-builtin-printf")
    set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS} -mcmodel=medany -march=rv64gcv_zfh -mabi=lp64d")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Map=output.map -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE} -T ${CMAKE_SOURCE_DIR}/htif.ld")
endif ()


target_compile_options(fast-depth PRIVATE -u _printf_float)


message("Compiler Flags: ${CMAKE_C_FLAGS}")
message("Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}")


add_subdirectory(../../nn ./build/nn)
target_link_libraries(fast-depth PUBLIC nn)

target_link_libraries(fast-depth PUBLIC m)

