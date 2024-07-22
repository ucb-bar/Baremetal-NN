
.globl NN__dot_f32
NN__dot_f32:
  vsetvli     t1,zero,e32,m1,ta,ma
  vmv.v.i     v27,0
  vmv1r.v     v24,v27
  beqz        a0,__dot_f32_exit
  slli        a3,a3,0x2
  slli        a5,a5,0x2
__dot_f32_loop:
  vsetvli     a6,a0,e32,m1,ta,ma
  vlse32.v    v26,(a2),a3
  vlse32.v    v25,(a4),a5
  vfmacc.vv   v24,v26,v25
  slli        a7,a6,0x2
  add         a2,a2,a7
  add         a4,a4,a7
  sub         a0,a0,a6
  bnez        a0,__dot_f32_loop
  vsetvli     t1,zero,e32,m1,ta,ma
__dot_f32_exit:
  vfredusum.vs  v24,v24,v27
  vfmv.f.s    fa5,v24
  fsw         fa5,0(a1)
  ret