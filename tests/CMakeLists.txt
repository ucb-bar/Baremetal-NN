project(test LANGUAGES C)

add_executable(test src/generated.c)
target_include_directories(test PUBLIC inc)

target_compile_features(test INTERFACE c_std_11)

if (X86)
  target_link_libraries(test PUBLIC target-x86)
elseif (RISCV)
  target_link_libraries(test PUBLIC target-riscv)
endif()


target_link_libraries(test PUBLIC nn)
target_link_libraries(test PUBLIC m)

