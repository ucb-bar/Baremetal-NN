.section .text
.align 4
.globl NN_add_f16_asm
NN_add_f16_asm:
  beqz        a0, __add_f16_exit
__add_f16_loop:
  vsetvli     a5, a0, e16, m1, ta, ma
  vle16.v     v24, (a2)
  vle16.v     v25, (a3)
  vfadd.vv    v24, v24, v25
  vse16.v     v24, (a1)
  slli        a4, a5, 0x1
  add         a2, a2, a4
  add         a3, a3, a4
  add         a1, a1, a4
  sub         a0, a0, a5
  bnez        a0, __add_f16_loop
__add_f16_exit:
  ret

.section .text
.align 4
.globl NN_add_f32_asm
NN_add_f32_asm:
  beqz        a0, __add_f32_exit
__add_f32_loop:
  vsetvli     a5, a0, e32, m1, ta, ma
  vle32.v     v24, (a2)
  vle32.v     v25, (a3)
  vfadd.vv    v24, v24, v25
  vse32.v     v24, (a1)
  slli        a4, a5, 0x2
  add         a2, a2, a4
  add         a3, a3, a4
  add         a1, a1, a4
  sub         a0, a0, a5
  bnez        a0, __add_f32_loop
__add_f32_exit:
  ret