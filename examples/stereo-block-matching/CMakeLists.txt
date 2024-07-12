cmake_minimum_required(VERSION 3.10)

project(stereoBM LANGUAGES C)

add_executable(stereoBM main.c)

target_include_directories(stereoBM PUBLIC ./)
target_include_directories(stereoBM PUBLIC ../utility)


target_compile_features(stereoBM INTERFACE c_std_11)

if (X86)
  message("stereoBM: building for x86")
  target_link_libraries(stereoBM PUBLIC target-x86)
elseif (RISCV)
  message("stereoBM: building for RISC-V")
  target_link_libraries(stereoBM PUBLIC target-riscv)
endif ()


target_compile_options(stereoBM PRIVATE -O3 -Wall -Wextra)

target_link_libraries(stereoBM PUBLIC nn)
