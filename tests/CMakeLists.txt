cmake_minimum_required(VERSION 3.10)

project(tests LANGUAGES C)

# Add custom command to generate C code test pattern
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/generated.c
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/generate_test.py -O ${CMAKE_CURRENT_SOURCE_DIR}/src/generated.c
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating C code test pattern"
)

# Add the generated file as a dependency
add_executable(tests ${CMAKE_CURRENT_SOURCE_DIR}/src/generated.c)

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

# New target to run the test using Spike
add_custom_target(run_tests
  COMMAND spike --isa=rv64gcv_zicntr_zfh_zvfh $<TARGET_FILE:tests>
  DEPENDS tests
  COMMENT "Running test with Spike: $<TARGET_FILE:tests>"
)
