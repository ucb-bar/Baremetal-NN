.section .text
.align 4
.globl NN_linear_f32_asm
NN_linear_f32_asm:
  vsetvli	    t3, zero, e32, m1, ta, ma
  beqz	      a1, __nn_linear_f32_exit
  slli	      t4, a0, 0x2
  neg	        t4, t4
  mv	        a7, a5
  slli	      a1, a1, 0x2
  add	        t3, a5, a1
  vmv.v.i	    v27, 0
__nn_linear_f32_loop0:
  vmv.v.i	    v24, 0
  beqz	      a0, __nn_linear_f32_loop2
  mv	        a1, a0
__nn_linear_f32_loop1:
  vsetvli	    a5, a1, e32, m1, ta, ma
  vle32.v	  v26, (a3)
  vle32.v	  v25, (a4)
  vfmacc.vv	  v24, v26, v25
  slli	      a6, a5, 0x2
  add	        a3, a3, a6
  add	        a4, a4, a6
  sub	        a1, a1, a5
  bnez	      a1, __nn_linear_f32_loop1
  vsetvli	    t1, zero, e32, m1, ta, ma
__nn_linear_f32_loop2:
  vfredusum.vs v24, v24, v27
  vfmv.f.s	  fa4, v24
  flw 	      fa5, 0(a7)
  fadd.s	    fa5, fa5, fa4
  fsw	        fa5, 0(a2)
  add	        a3, a3, t4
  addi	      a7, a7, 4
  addi	      a2, a2, 4
  bne	        a7, t3, __nn_linear_f32_loop0
__nn_linear_f32_exit:
  ret