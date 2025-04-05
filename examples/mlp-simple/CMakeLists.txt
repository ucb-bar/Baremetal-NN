cmake_minimum_required(VERSION 3.10)

project(mlp-simple LANGUAGES C)

add_executable(mlp-simple main.c)

target_include_directories(mlp-simple PUBLIC ./)


target_compile_features(mlp-simple INTERFACE c_std_11)

if (X86)
    message("mlp-simple: building for x86")
    target_link_libraries(mlp-simple PUBLIC target-x86)
elseif (RISCV)
    message("mlp-simple: building for RISC-V")
    target_link_libraries(mlp-simple PUBLIC target-riscv)
    set_target_properties(mlp-simple PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(mlp-simple PRIVATE -O3 -Wall -Wextra)

target_link_libraries(mlp-simple PUBLIC nn)
target_link_libraries(mlp-simple PUBLIC m)
