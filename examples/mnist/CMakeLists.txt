cmake_minimum_required(VERSION 3.10)

project(mnist LANGUAGES C)

add_executable(mnist main.c)

target_include_directories(mnist PUBLIC ./)
target_include_directories(mnist PUBLIC ../utility)


target_compile_features(mnist INTERFACE c_std_11)

if (X86)
  message(STATUS "mnist: Building for x86")
  target_link_libraries(mnist PUBLIC target-x86)
elseif (RISCV)
  message(STATUS "mnist: Building for RISC-V")
  target_link_libraries(mnist PUBLIC target-riscv)
  set_target_properties(mnist PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(mnist PRIVATE -O3 -Wall -Wextra)

target_link_libraries(mnist PUBLIC nn)
target_link_libraries(mnist PUBLIC m)
