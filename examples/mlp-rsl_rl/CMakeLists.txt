cmake_minimum_required(VERSION 3.10)

project(mlp-rsl_rl LANGUAGES C)

add_executable(mlp-rsl_rl main.c)

target_include_directories(mlp-rsl_rl PUBLIC ./)
target_include_directories(mlp-rsl_rl PUBLIC ../utility)


target_compile_features(mlp-rsl_rl INTERFACE c_std_11)

if (X86)
  message(STATUS "mlp-rsl_rl: Building for x86")
  target_link_libraries(mlp-rsl_rl PUBLIC target-x86)
elseif (RISCV)
  message(STATUS "mlp-rsl_rl: Building for RISC-V")
  target_link_libraries(mlp-rsl_rl PUBLIC target-riscv)
  set_target_properties(mlp-rsl_rl PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(mlp-rsl_rl PRIVATE -O3 -Wall -Wextra)

target_link_libraries(mlp-rsl_rl PUBLIC nn)
target_link_libraries(mlp-rsl_rl PUBLIC m)
