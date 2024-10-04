
/**
  * @file glossy-htif.ld
  * @brief Linker script to describe the memory layout of a generic RISC-V platform with HTIF interface.
  * It defines following symbols, which code can use without definition:
  *   __boot_hart
  *   __stack_size
  *   __heap_size
  *   __text_start
  *   __text_load_start
  *   __text_end
  *   __data_start
  *   __data_load_start
  *   __preinit_array_start
  *   __preinit_array_end
  *   __init_array_start
  *   __init_array_end
  *   __fini_array_start
  *   __fini_array_end
  *   __tdata_start
  *   __tdata_end
  *   __tdata_size
  *   __data_end
  *   __tbss_start
  *   __tbss_end
  *   __tbss_size
  *   __bss_start
  *   __bss_end
  *   __end
  *   __heap_start
  *   __heap_end
  *   __stack_start
  *   __sp
  *   __stack_end
  *   __stack_shift
  *
  */

OUTPUT_ARCH("riscv")

/* Program entry point */
ENTRY(_start)

MEMORY {
  DRAM     (rwx): ORIGIN = 0x80000000, LENGTH = 16384M
}

SECTIONS {
  /* Default boot hart */
  PROVIDE(__boot_hart = 0);

  /* Default stack size */
  __stack_size = DEFINED(__stack_size) ? __stack_size : 24K;
  PROVIDE(__stack_size = __stack_size);

  /* Default heap size */
  __heap_size = DEFINED(__heap_size) ? __heap_size : 128K;
  PROVIDE(__heap_size = __heap_size);
  
  /* Section containing the program */
  .text : ALIGN(4) {
    PROVIDE(__text_start = .);
    *(.text.init)
    *(.text.trap_vector)
    *(.text.unlikely .text.unlikely.*)
    *(.text.startup .text.startup.*)
    *(.text .text.*)
    *(.gnu.linkonce.t.*)
    PROVIDE(__text_end = .);
  }> DRAM
  PROVIDE(__text_load_start = LOADADDR(.text));

  /* Section containing read-only data */
  .rodata : {
    . = ALIGN(4);
    *(.rodata .rodata.*)
    *(.gnu.linkonce.r.*)
    . = ALIGN(8);
    *(.srodata.cst16)
    *(.srodata.cst8)
    *(.srodata.cst4)
    *(.srodata.cst2)
    *(.srodata .srodata.*)
  }> DRAM

  /* Section for initialized data */
  .data : ALIGN(8) {
    PROVIDE(__data_start = .);
    *(.data)
    *(.gnu.linkonce.d.*)
  }> DRAM
  
  PROVIDE(__data_load_start = LOADADDR(.data));

  .preinit_array : {
    PROVIDE_HIDDEN(__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN(__preinit_array_end = .);
  }> DRAM
  
  .init_array : {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
    KEEP(*(.init_array EXCLUDE_FILE(*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .ctors))
    PROVIDE_HIDDEN(__init_array_end = .);
  }> DRAM

  .fini_array : {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
    KEEP(*(.fini_array EXCLUDE_FILE(*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .dtors))
    PROVIDE_HIDDEN(__fini_array_end = .);
  }> DRAM

  .ctors : {
    KEEP(*crtbegin.o(.ctors))
    KEEP(*crtbegin?.o(.ctors))
    KEEP(*(EXCLUDE_FILE(*crtend.o *crtend?.o) .ctors))
    KEEP(*(SORT(.ctors.*)))
    KEEP(*(.ctors))
  }> DRAM

  .dtors : {
    KEEP(*crtbegin.o(.dtors))
    KEEP(*crtbegin?.o(.dtors))
    KEEP(*(EXCLUDE_FILE(*crtend.o *crtend?.o) .dtors))
    KEEP(*(SORT(.dtors.*)))
    KEEP(*(.dtors))
  }> DRAM

  /* Section for initialized small data */
  .sdata : {
    PROVIDE(__global_pointer$ = . + 0x800);
    *(.sdata .sdata.*)
    *(.gnu.linkonce.s.*)
  }> DRAM

  /* Section for initialized thread-local small data */
  .tdata : {
    PROVIDE_HIDDEN(__tdata_start = .);
    *(.tdata .tdata.*)
    *(.gnu.linkonce.td.*)
    PROVIDE_HIDDEN(__tdata_end = .);
    PROVIDE(__data_end = .);
  }> DRAM
  PROVIDE(__tdata_size = SIZEOF(.tdata));

  /* Section for zero-initialized thread-local uninitialized data */
  .tbss (NOLOAD) : ALIGN(8) {
    PROVIDE_HIDDEN(__tbss_start = .);
    PROVIDE_HIDDEN(__tbss_offset = ABSOLUTE(__tbss_start - __tdata_start));
    *(.tbss .tbss.*)
    *(.gnu.linkonce.tb.*)
    *(.tcommon)
    PROVIDE_HIDDEN(__tbss_end = .);
  }> DRAM
  PROVIDE(__tbss_size = SIZEOF(.tbss));
  
  /* Section for uninitialized data */
  .bss : ALIGN (8) {
    PROVIDE_HIDDEN(__bss_start = .);
    *(.sbss .sbss.*)
    *(.gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(.gnu.linkonce.b.*)
    *(COMMON)
    PROVIDE_HIDDEN(__bss_end = .);
  }> DRAM

  . = ALIGN (8);
  PROVIDE(__end = .);

  /* Reserve heap space */
  .heap (NOLOAD) : ALIGN(8) {
    PROVIDE_HIDDEN(__heap_start = .);
    . += __heap_size;
    . = ALIGN(4K);
    PROVIDE_HIDDEN(__heap_end = .);
  }> DRAM

  /* Reserve stack space */
  .stack (NOLOAD) : ALIGN(16) {
    PROVIDE_HIDDEN(__stack_start = .);
    . += __stack_size;
    PROVIDE(__sp = .);
    PROVIDE_HIDDEN(__stack_end = .);
  }> DRAM

  PROVIDE(__stack_shift = LOG2CEIL(__stack_size));
  
  /* C++ exception handling information is
   * not useful with our current runtime environment,
   * and it consumes flash space. Discard it until
   * we have something that can use it
   */
  /DISCARD/ : {
    *(.eh_frame .eh_frame.*)
  }
}
