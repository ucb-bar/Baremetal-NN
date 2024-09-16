
.globl NN__add1_f32
NN__add1_f32:
  beqz        a0,__add1_f32_exit
  slli        a4,a4,0x2
  slli        a2,a2,0x2
__add1_f32_loop:
  vsetvli     a5,a0,e32,m1,ta,ma
  vlse32.v    v24,(a3),a4
  vfmv.v.f    v25,fa0
  vfadd.vv    v24,v24,v25
  vsse32.v    v24,(a1),a2
  slli        a6,a5,0x2
  add         a3,a3,a6
  add         a1,a1,a6
  sub         a0,a0,a5
  bnez        a0,__add1_f32_loop
__add1_f32_exit:
  ret