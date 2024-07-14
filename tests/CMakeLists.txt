project(tests LANGUAGES C)

add_executable(tests src/generated.c)
target_include_directories(tests PUBLIC inc)

target_compile_features(tests INTERFACE c_std_11)

if (X86)
  target_link_libraries(tests PUBLIC target-x86)
elseif (RISCV)
  target_link_libraries(tests PUBLIC target-riscv)
endif()


# include_directories(
#   ../nn
#   ../nn/functional 
#   ../nn/impl)

# find_library(LIB_TO_INCLUDE nn ./)

# target_link_libraries(tests PUBLIC ${LIB_TO_INCLUDE})

target_link_libraries(tests PUBLIC nn)
target_link_libraries(tests PUBLIC m)

