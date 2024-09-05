cmake_minimum_required(VERSION 3.10)

project(diffuse-loco LANGUAGES C)

add_executable(diffuse-loco main.c)

target_include_directories(diffuse-loco PUBLIC ./)


target_compile_features(diffuse-loco INTERFACE c_std_11)

if (X86)
  message("diffuse-loco: building for x86")
  target_link_libraries(diffuse-loco PUBLIC target-x86)
elseif (RISCV)
  message("diffuse-loco: building for RISC-V")
  target_link_libraries(diffuse-loco PUBLIC target-riscv)
  set_target_properties(diffuse-loco PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(diffuse-loco -O3 -Wall -Wextra)

target_link_libraries(diffuse-loco PUBLIC nn)
target_link_libraries(diffuse-loco PUBLIC m)
