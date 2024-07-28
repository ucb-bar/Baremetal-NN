cmake_minimum_required(VERSION 3.10)

project(tests LANGUAGES C)

add_executable(tests src/generated.c)
target_include_directories(tests PUBLIC inc)

target_compile_features(tests INTERFACE c_std_11)

if (X86)
  target_link_libraries(tests PUBLIC target-x86)
elseif (RISCV)
  target_link_libraries(tests PUBLIC target-riscv)
  set_target_properties(tests PROPERTIES SUFFIX ".elf")
endif()


target_link_libraries(tests PUBLIC nn)
target_link_libraries(tests PUBLIC m)

