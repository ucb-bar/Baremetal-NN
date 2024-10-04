.section .text
.align 4
.globl NN_addmm_f16_asm
NN_addmm_f16_asm:
  vsetvli	    t3, zero, e16, m1, ta, ma
  beqz	      a1, __nn_addmm_f16_exit
  slli	      t4, a0, 0x1
  neg	        t4, t4
  mv	        a7, a5
  slli	      a1, a1, 0x1
  add	        t3, a5, a1
  vmv.v.i	    v27, 0
__nn_addmm_f16_loop0:
  vmv.v.i	    v24, 0
  beqz	      a0, __nn_addmm_f16_loop2
  mv	        a1, a0
__nn_addmm_f16_loop1:
  vsetvli	    a5, a1, e16, m1, ta, ma
  vle16.v	  v26, (a3)
  vle16.v	  v25, (a4)
  vfmacc.vv	  v24, v26, v25
  slli	      a6, a5, 0x1
  add	        a3, a3, a6
  add	        a4, a4, a6
  sub	        a1, a1, a5
  bnez	      a1, __nn_addmm_f16_loop1
  vsetvli	    t1, zero, e16, m1, ta, ma
__nn_addmm_f16_loop2:
  #ifdef DEBUG_USE_REDOSUM
    vfredosum.vs v24, v24, v27
  #else
    vfredusum.vs v24, v24, v27
  #endif
  vfmv.f.s	  fa4, v24
  flh 	      fa5, 0(a7)
  fadd.h	    fa5, fa5, fa4
  fsh	        fa5, 0(a2)
  add	        a3, a3, t4
  addi	      a7, a7, 2
  addi	      a2, a2, 2
  bne	        a7, t3, __nn_addmm_f16_loop0
__nn_addmm_f16_exit:
  ret

.section .text
.align 4
.globl NN_addmm_f32_asm
NN_addmm_f32_asm:
  vsetvli	    t3, zero, e32, m1, ta, ma
  beqz	      a1, __nn_addmm_f32_exit
  slli	      t4, a0, 0x2
  neg	        t4, t4
  mv	        a7, a5
  slli	      a1, a1, 0x2
  add	        t3, a5, a1
  vmv.v.i	    v27, 0
__nn_addmm_f32_loop0:
  vmv.v.i	    v24, 0
  beqz	      a0, __nn_addmm_f32_loop2
  mv	        a1, a0
__nn_addmm_f32_loop1:
  vsetvli	    a5, a1, e32, m1, ta, ma
  vle32.v	  v26, (a3)
  vle32.v	  v25, (a4)
  vfmacc.vv	  v24, v26, v25
  slli	      a6, a5, 0x2
  add	        a3, a3, a6
  add	        a4, a4, a6
  sub	        a1, a1, a5
  bnez	      a1, __nn_addmm_f32_loop1
  vsetvli	    t1, zero, e32, m1, ta, ma
__nn_addmm_f32_loop2:
  #ifdef DEBUG_USE_REDOSUM
    vfredosum.vs v24, v24, v27
  #else
    vfredusum.vs v24, v24, v27
  #endif
  vfmv.f.s	  fa4, v24
  flw 	      fa5, 0(a7)
  fadd.s	    fa5, fa5, fa4
  fsw	        fa5, 0(a2)
  add	        a3, a3, t4
  addi	      a7, a7, 4
  addi	      a2, a2, 4
  bne	        a7, t3, __nn_addmm_f32_loop0
__nn_addmm_f32_exit:
  ret
