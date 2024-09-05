cmake_minimum_required(VERSION 3.10)

project(mlp-ppo LANGUAGES C)

add_executable(mlp-ppo main.c)

target_include_directories(mlp-ppo PUBLIC ./)
target_include_directories(mlp-ppo PUBLIC ../utility)


target_compile_features(mlp-ppo INTERFACE c_std_11)

if (X86)
  message(STATUS "mlp-ppo: Building for x86")
  target_link_libraries(mlp-ppo PUBLIC target-x86)
elseif (RISCV)
  message(STATUS "mlp-ppo: Building for RISC-V")
  target_link_libraries(mlp-ppo PUBLIC target-riscv)
  set_target_properties(mlp-ppo PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(mlp-ppo PRIVATE -O3 -Wall -Wextra)

target_link_libraries(mlp-ppo PUBLIC nn)
target_link_libraries(mlp-ppo PUBLIC m)
