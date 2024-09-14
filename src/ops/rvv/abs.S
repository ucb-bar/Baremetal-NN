
.globl NN__abs_f32
NN__abs_f32:
  beqz        a0,__abs_f32_exit
  slli        a4,a4,0x2
  slli        a2,a2,0x2
__abs_f32_loop:
  vsetvli     a5,a0,e32,m1,ta,ma
  vlse32.v    v24,(a3),a4
  vfabs.v     v24,v24
  vsse32.v    v24,(a1),a2
  slli        a6,a5,0x2
  add         a3,a3,a6
  add         a1,a1,a6
  sub         a0,a0,a5
  bnez        a0,__abs_f32_loop
__abs_f32_exit:
  ret