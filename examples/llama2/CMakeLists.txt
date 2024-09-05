cmake_minimum_required(VERSION 3.10)

project(llama2 LANGUAGES C)

add_executable(llama2 main.c)

target_include_directories(llama2 PUBLIC ./)


target_compile_features(llama2 INTERFACE c_std_11)

if (X86)
  message(STATUS "llama2: Building for x86")
elseif (RISCV)
  message(STATUS "llama2: Building for RISC-V")
  target_link_libraries(llama2 PUBLIC target-riscv)
  set_target_properties(llama2 PROPERTIES SUFFIX ".elf")
endif ()

target_compile_options(llama2 PRIVATE -O3 -Wall -Wextra)

target_link_libraries(llama2 PUBLIC nn)
target_link_libraries(llama2 PUBLIC m)


add_custom_target(llama2-dump ALL
  COMMAND ${CMAKE_OBJDUMP} -d $<TARGET_FILE:llama2> > ${CMAKE_BINARY_DIR}/llama2.disassemble.S
  COMMAND ${CMAKE_OBJDUMP} -h $<TARGET_FILE:llama2> > ${CMAKE_BINARY_DIR}/llama2.sections
  COMMAND ${CMAKE_OBJDUMP} -t $<TARGET_FILE:llama2> > ${CMAKE_BINARY_DIR}/llama2.symbols
  DEPENDS llama2
  COMMENT "Generating dump files for llama2:\n ${CMAKE_BINARY_DIR}/llama2.disassemble.S\n ${CMAKE_BINARY_DIR}/llama2.sections\n ${CMAKE_BINARY_DIR}/llama2.symbols"
)
