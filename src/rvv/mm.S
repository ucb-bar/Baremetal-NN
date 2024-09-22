.section .text
.align 4
.globl NN_mm_f16_asm
NN_mm_f16_asm:
  vsetvli	    a7, zero, e16, m1, ta, ma
  beqz	      a1, __nn_mm_f16_exit
  slli	      t3, a0, 0x1
  neg	        t3, t3
  mv	        a6, a2
  slli	      a1, a1, 0x1
  add	        t1, a2, a1
  vmv.v.i	    v27, 0
__nn_mm_f16_loop0:
  vmv.v.i	    v24, 0
  beqz	      a0, __nn_mm_f16_loop2
  mv	        a2, a0
__nn_mm_f16_loop1:
  vsetvli	    a5, a2, e16, m1, ta, ma
  vle16.v	    v26, (a3)
  vle16.v	    v25, (a4)
  vfmacc.vv	  v24, v26, v25
  slli	      a1, a5, 0x1
  add	        a3, a3, a1
  add	        a4, a4, a1
  sub	        a2, a2, a5
  bnez	      a2, __nn_mm_f16_loop1
  vsetvli	    a7, zero, e16, m1, ta, ma
__nn_mm_f16_loop2:
  vfredusum.vs v24, v24, v27
  vfmv.f.s	  fa5, v24
  fsh	        fa5, 0(a6)
  add	        a3, a3, t3
  addi	      a6, a6, 2
  bne	        a6, t1, __nn_mm_f16_loop0
__nn_mm_f16_exit:
  ret

.section .text
.align 4
.globl NN_mm_f32_asm
NN_mm_f32_asm:
  vsetvli	    a7, zero, e32, m1, ta, ma
  beqz	      a1, __nn_mm_f32_exit
  slli	      t3, a0, 0x2
  neg	        t3, t3
  mv	        a6, a2
  slli	      a1, a1, 0x2
  add	        t1, a2, a1
  vmv.v.i	    v27, 0
__nn_mm_f32_loop0:
  vmv.v.i	    v24, 0
  beqz	      a0, __nn_mm_f32_loop2
  mv	        a2, a0
__nn_mm_f32_loop1:
  vsetvli	    a5, a2, e32, m1, ta, ma
  vle32.v	    v26, (a3)
  vle32.v	    v25, (a4)
  vfmacc.vv	  v24, v26, v25
  slli	      a1, a5, 0x2
  add	        a3, a3, a1
  add	        a4, a4, a1
  sub	        a2, a2, a5
  bnez	      a2, __nn_mm_f32_loop1
  vsetvli	    a7, zero, e32, m1, ta, ma
__nn_mm_f32_loop2:
  vfredusum.vs v24, v24, v27
  vfmv.f.s	  fa5, v24
  fsw	        fa5, 0(a6)
  add	        a3, a3, t3
  addi	      a6, a6, 4
  bne	        a6, t1, __nn_mm_f32_loop0
__nn_mm_f32_exit:
  ret
