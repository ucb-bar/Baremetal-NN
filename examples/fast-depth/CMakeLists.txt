cmake_minimum_required(VERSION 3.10)

project(fast-depth LANGUAGES C)

add_executable(fast-depth main.c)

target_include_directories(fast-depth PUBLIC ./)
target_include_directories(fast-depth PUBLIC ../utility)


target_compile_features(fast-depth INTERFACE c_std_11)

if (X86)
  message(STATUS "fast-depth: Building for x86")
  target_link_libraries(fast-depth PUBLIC target-x86)
elseif (RISCV)
  message(STATUS "fast-depth: Building for RISC-V")
  target_link_libraries(fast-depth PUBLIC target-riscv)
  set_target_properties(fast-depth PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(fast-depth PRIVATE -O3 -Wall -Wextra)

target_link_libraries(fast-depth PUBLIC nn)
target_link_libraries(fast-depth PUBLIC m)
