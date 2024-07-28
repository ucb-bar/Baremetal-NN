cmake_minimum_required(VERSION 3.10)

project(mlp LANGUAGES C)

add_executable(mlp main.c)

target_include_directories(mlp PUBLIC ./)


target_compile_features(mlp INTERFACE c_std_11)

if (X86)
    message("mlp: building for x86")
    target_link_libraries(mlp PUBLIC target-x86)
elseif (RISCV)
    message("mlp: building for RISC-V")
    target_link_libraries(mlp PUBLIC target-riscv)
    set_target_properties(mlp PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(mlp PRIVATE -O3 -Wall -Wextra)

target_link_libraries(mlp PUBLIC nn)
target_link_libraries(mlp PUBLIC m)
