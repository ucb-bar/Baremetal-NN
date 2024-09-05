cmake_minimum_required(VERSION 3.10)

project(simple LANGUAGES C)

add_executable(simple main.c)

target_include_directories(simple PUBLIC ./)


target_compile_features(simple INTERFACE c_std_11)

if (X86)
  message(STATUS "simple: Building for x86")
  target_link_libraries(simple PUBLIC target-x86)
elseif (RISCV)
  message(STATUS "simple: Building for RISC-V")
  target_link_libraries(simple PUBLIC target-riscv)
  set_target_properties(simple PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(simple PRIVATE -O3 -Wall -Wextra)

target_link_libraries(simple PUBLIC nn)
target_link_libraries(simple PUBLIC m)
