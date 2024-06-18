
# Options
option(BUILD_SHARED_LIBS    "Build using shared libraries"      OFF     )
option(CMAKE_BUILD_TYPE     "Build type"                        Debug   )
option(STATIC_LINKING       "Link statically"                   TRUE    )

option(X86                  "Build for x86"                     OFF     )
option(RISCV                "Build for RISC-V"                  OFF     )


add_library(target-riscv INTERFACE)
target_compile_features(target-riscv INTERFACE c_std_11)
target_compile_definitions(target-riscv INTERFACE RISCV)

add_library(target-x86 INTERFACE)
target_compile_features(target-x86 INTERFACE c_std_11)
target_compile_definitions(target-x86 INTERFACE X86)

target_link_options(target-x86 INTERFACE -static)
target_link_options(target-riscv INTERFACE -static)

target_compile_options(target-x86 INTERFACE -march=native)
target_link_options(target-x86 INTERFACE -march=native)


set(WRAP_SPECS_FILE     "htif_wrap.specs")
set(SPECS_FILE          "htif_nano.specs")
set(LIBGLOSS_DIR        "$ENV{RISCV}/riscv64-unknown-elf/lib/")

target_compile_options(target-riscv INTERFACE -fno-common -fno-builtin-printf)
target_compile_options(target-riscv INTERFACE -mcmodel=medany -march=rv64gcv_zfh -mabi=lp64d)
target_compile_options(target-riscv INTERFACE -Wl,-Map=output.map -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE} -T ${CMAKE_SOURCE_DIR}/htif.ld)
target_link_options(target-x86 INTERFACE -fno-common -fno-builtin-printf)
target_link_options(target-riscv INTERFACE -mcmodel=medany -march=rv64gcv_zfh -mabi=lp64d)
target_link_options(target-riscv INTERFACE -Wl,-Map=output.map -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE} -T ${CMAKE_SOURCE_DIR}/htif.ld)

