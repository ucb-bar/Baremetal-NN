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


add_custom_target(dump ALL
  COMMAND ${CMAKE_OBJDUMP} -d $<TARGET_FILE:tests> > ${CMAKE_BINARY_DIR}/tests.disassemble.S
  COMMAND ${CMAKE_OBJDUMP} -h $<TARGET_FILE:tests> > ${CMAKE_BINARY_DIR}/tests.sections
  COMMAND ${CMAKE_OBJDUMP} -t $<TARGET_FILE:tests> > ${CMAKE_BINARY_DIR}/tests.symbols
  DEPENDS tests
  COMMENT "Generating dump files for tests:\n ${CMAKE_BINARY_DIR}/tests.disassemble.S\n ${CMAKE_BINARY_DIR}/tests.sections\n ${CMAKE_BINARY_DIR}/tests.symbols"
)
