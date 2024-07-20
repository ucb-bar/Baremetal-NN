cmake_minimum_required(VERSION 3.10)

project(cartpole LANGUAGES C)

add_executable(cartpole main.c)

target_include_directories(cartpole PUBLIC ./)


target_compile_features(cartpole INTERFACE c_std_11)

if (X86)
    message("cartpole: building for x86")
    target_link_libraries(cartpole PUBLIC target-x86)
elseif (RISCV)
    message("cartpole: building for RISC-V")
    target_link_libraries(cartpole PUBLIC target-riscv)
endif ()

target_compile_options(cartpole PRIVATE -O2 -Wall -Wextra)

target_link_libraries(cartpole PUBLIC nn)
target_link_libraries(cartpole PUBLIC m)
