
.globl NN__add_f32
NN__add_f32:
  beqz        a0,__add_f32_exit
  slli        a4,a4,0x2
  slli        a6,a6,0x2
  slli        a2,a2,0x2
__add_f32_loop:
  vsetvli     a7,a0,e32,m1,ta,ma
  vlse32.v    v24,(a3),a4
  vlse32.v    v25,(a5),a6
  vfadd.vv    v24,v24,v25
  vsse32.v    v24,(a1),a2
  slli        t1,a7,0x2
  add         a3,a3,t1
  add         a5,a5,t1
  add         a1,a1,t1
  sub         a0,a0,a7
  bnez        a0,__add_f32_loop
__add_f32_exit:
  ret