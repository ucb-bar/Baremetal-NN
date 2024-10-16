cmake_minimum_required(VERSION 3.10)

project(mlp-flappy LANGUAGES C)

add_executable(mlp-flappy main.c)

target_include_directories(mlp-flappy PUBLIC ./)


target_compile_features(mlp-flappy INTERFACE c_std_11)

if (X86)
    message("mlp-flappy: building for x86")
    target_link_libraries(mlp-flappy PUBLIC target-x86)
elseif (RISCV)
    message("mlp-flappy: building for RISC-V")
    target_link_libraries(mlp-flappy PUBLIC target-riscv)
    set_target_properties(mlp-flappy PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(mlp-flappy PRIVATE -O3 -Wall -Wextra)

target_link_libraries(mlp-flappy PUBLIC nn)
target_link_libraries(mlp-flappy PUBLIC m)
