/*****************************************************************************/
/* Copyright YouDao, Inc.                                                    */
/*                                                                           */
/* Licensed under the Apache License, Version 2.0 (the "License");           */
/* you may not use this file except in compliance with the License.          */
/* You may obtain a copy of the License at                                   */
/*                                                                           */
/*     http://www.apache.org/licenses/LICENSE-2.0                            */
/*                                                                           */
/* Unless required by applicable law or agreed to in writing, software       */
/* distributed under the License is distributed on an "AS IS" BASIS,         */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  */
/* See the License for the specific language governing permissions and       */
/* limitations under the License.                                            */
/*****************************************************************************/


#include <stdint.h>
#include <arm_neon.h>

#ifndef INCLUDE_A35_KERNEL
#define INCLUDE_A35_KERNEL

/* for cortex-a35, fp32 NEON operation on q regs are not recommended,
 * using d regs without broadcast is better */

/* for cortex-a35 fp32 fma instruction sequence,
 * it's recommended to put 3 nearest fma inst together */
#define FMA_3V(c1, c2, c3, a1, a2, a3, b1, b2, b3) \
  "fmla v"#c1".2s,v"#a1".2s,v"#b1".2s\n\t"\
  "fmla v"#c2".2s,v"#a2".2s,v"#b2".2s\n\t"\
  "fmla v"#c3".2s,v"#a3".2s,v"#b3".2s\n\t"

#define INIT_3V(c1, c2, c3) \
  "movi v"#c1".8b,#0; movi v"#c2".8b,#0; movi v"#c3".8b,#0\n\t"

#define INIT_4V(c1, c2, c3, c4) INIT_3V(c1, c2, c3)\
  "movi v"#c4".8b,#0\n\t"

/* x12 - x15 for c_tmp pointers */
/* v0 always for beta at storage status */

#define INIT_SAVE_M3_CR \
  "ld1r {v0.2s},[%[beta_addr]]\n\t"\
  "mov x12,%[c_ptr]; add x13,%[c_ptr],%w[LDC],UXTW #2\n\t"\
  "add x14,%[c_ptr],%w[LDC],UXTW #3\n\t"

#define INIT_SAVE_M4_CR INIT_SAVE_M3_CR \
  "add x15,x13,%w[LDC],UXTW #3\n\t"

#define INIT_SAVE_CC \
  "ld1r {v0.2s},[%[beta_addr]]; mov x12,%[c_ptr]\n\t"\
  "add x13,%[c_ptr],%w[LDC],UXTW #2\n\t"

/* c1[0], c1[1] */
/* c2[0], c2[1] */
/* c3[0], c3[1] */
/* c4[0], c4[1] */
/* clobber: x12 - x13, v0 - v4 */
#define UNIT_SAVE_M4N2_CC(c1, c2, c3, c4) \
  "ldr d1,[x12]; ldr d2,[x12,#8]\n\t"\
  "trn1 v3.2s,v"#c1".2s,v"#c2".2s; trn1 v4.2s,v"#c3".2s,v"#c4".2s\n\t"\
  "trn2 v"#c2".2s,v"#c1".2s,v"#c2".2s; trn2 v"#c4".2s,v"#c3".2s,v"#c4".2s\n\t"\
  "ldr d"#c1",[x13]; ldr d"#c3",[x13,#8]\n\t"\
  "fmla v3.2s,v1.2s,v0.2s\n\t"\
  "fmla v4.2s,v2.2s,v0.2s\n\t"\
  "fmla v"#c2".2s,v"#c1".2s,v0.2s\n\t"\
  "fmla v"#c4".2s,v"#c3".2s,v0.2s\n\t"\
  "str d3,[x12]; str d4,[x12,#8]\n\t"\
  "prfm pstl2keep,[x12,#32]; add x12,x12,%w[LDC],UXTW #3\n\t"\
  "str d"#c2",[x13]; str d"#c4",[x13,#8]\n\t"\
  "prfm pstl2keep,[x13,#32]; add x13,x13,%w[LDC],UXTW #3\n\t"

/* clobber: x12 - x15, v0 - v4 */
#define UNIT_SAVE_M4N2_CR(c1, c2, c3, c4) \
  "ldr d1,[x12]; ldr d2,[x13]\n\t"\
  "ldr d3,[x14]; ldr d4,[x15]\n\t"\
  "fmla v"#c1".2s,v1.2s,v0.2s\n\t"\
  "fmla v"#c2".2s,v2.2s,v0.2s\n\t"\
  "fmla v"#c3".2s,v3.2s,v0.2s\n\t"\
  "fmla v"#c4".2s,v4.2s,v0.2s\n\t"\
  "str d"#c1",[x12],#8; str d"#c2",[x13],#8\n\t"\
  "str d"#c3",[x14],#8; str d"#c4",[x15],#8\n\t"

/* c1[0], c1[1] */
/* c2[0], c2[1] */
/* c3[0], c3[1] */
/* clobber: x12 - x13, v0 - v3 */
#define UNIT_SAVE_M3N2_CC(c1, c2, c3) \
  "ldr d1,[x12]\n\t"\
  "trn1 v2.2s,v"#c1".2s,v"#c2".2s\n\t"\
  "trn2 v"#c2".2s,v"#c1".2s,v"#c2".2s\n\t"\
  "ldr d"#c1",[x13]; ldr s3,[x12,#8]\n\t"\
  "fmla v2.2s,v1.2s,v0.2s\n\t"\
  "fmla v"#c2".2s,v"#c1".2s,v0.2s\n\t"\
  "ldr s1,[x13,#8]\n\t"\
  "str d2,[x12]; ins v2.s[0],v"#c3".s[1]\n\t"\
  "str d"#c2",[x13]\n\t"\
  "fmla s"#c3",s3,v0.s[0]; fmla s2,s1,v0.s[0]\n\t"\
  "str s"#c3",[x12,#8]; prfm pstl2keep,[x12,#24]\n\t"\
  "add x12,x12,%w[LDC],UXTW #3\n\t"\
  "str s2,[x13,#8]; prfm pstl2keep,[x13,#24]\n\t"\
  "add x13,x13,%w[LDC],UXTW #3\n\t"

/* clobber: x12 - x14, v0 - v3 */
#define UNIT_SAVE_M3N2_CR(c1, c2, c3) \
  "ldr d1,[x12]; ldr d2,[x13]\n\t"\
  "ldr d3,[x14]\n\t"\
  "fmla v"#c1".2s,v1.2s,v0.2s\n\t"\
  "fmla v"#c2".2s,v2.2s,v0.2s\n\t"\
  "fmla v"#c3".2s,v3.2s,v0.2s\n\t"\
  "str d"#c1",[x12],#8; str d"#c2",[x13],#8\n\t"\
  "str d"#c3",[x14],#8\n\t"

/* c1[0] + c1[1] */
/* c2[0] + c2[1] */
/* c3[0] + c3[1] */
/* c4[0] + c4[1] */
/* clobber: x12, v0 - v4 */
#define UNIT_SAVE_M4N1_CC(c1, c2, c3, c4) \
  "ldr d3,[x12]; ldr d4,[x12,#8]\n\t"\
  "faddp v1.2s,v"#c1".2s,v"#c2".2s\n\t"\
  "faddp v2.2s,v"#c3".2s,v"#c4".2s\n\t"\
  "fmla v1.2s,v3.2s,v0.2s\n\t"\
  "fmla v2.2s,v4.2s,v0.2s\n\t"\
  "str d1,[x12]; str d2,[x12,#8]\n\t"\
  "prfm pstl2keep,[x12,#32]\n\t"\
  "add x12,x12,%w[LDC],UXTW #2\n\t"

/* clobber: x12 - x15, v0 - v4 */
#define UNIT_SAVE_M4N1_CR(c1, c2, c3, c4) \
  "ldr s1,[x12]; ldr s2,[x13]\n\t"\
  "faddp v"#c1".2s,v"#c1".2s,v"#c3".2s\n\t"\
  "ld1 {v1.s}[1],[x14]; ld1 {v2.s}[1],[x15]\n\t"\
  "faddp v"#c2".2s,v"#c2".2s,v"#c4".2s\n\t"\
  "fmla v"#c1".2s,v1.2s,v0.2s\n\t"\
  "fmla v"#c2".2s,v2.2s,v0.2s\n\t"\
  "str s"#c1",[x12],#4; str s"#c2",[x13],#4\n\t"\
  "st1 {v"#c1".s}[1],[x14],#4; st1 {v"#c2".s}[1],[x15],#4\n\t"

/* c1[0] + c1[1] */
/* c2[0] + c2[1] */
/* c3[0] + c3[1] */
/* clobber: x12, v0 - v3 */
#define UNIT_SAVE_M3N1_CC(c1, c2, c3) \
  "ldr d1,[x12]; ldr s2,[x12,#8]\n\t"\
  "faddp v"#c1".2s,v"#c1".2s,v"#c2".2s\n\t"\
  "faddp s"#c3",v"#c3".2s\n\t"\
  "fmla v"#c1".2s,v1.2s,v0.2s\n\t"\
  "fmla s"#c3",s2,v0.s[0]\n\t"\
  "str d"#c1",[x12]; str s"#c3",[x12,#8]\n\t"\
  "prfm pstl2keep,[x12,#24]\n\t"\
  "add x12,x12,%w[LDC],UXTW #2\n\t"

/* clobber: x12 - x14, v0 - v3 */
#define UNIT_SAVE_M3N1_CR(c1, c2, c3) \
  "ldr s1,[x12]; ldr s2,[x13]; ldr s3,[x14]\n\t"\
  "faddp s"#c1",v"#c1".2s; faddp s"#c2",v"#c2".2s; faddp s"#c3",v"#c3".2s\n\t"\
  "fmla s"#c1",s1,v0.s[0]\n\t"\
  "fmla s"#c2",s2,v0.s[0]\n\t"\
  "fmla s"#c3",s3,v0.s[0]\n\t"\
  "str s"#c1",[x12],#4\n\t"\
  "str s"#c2",[x13],#4\n\t"\
  "str s"#c3",[x14],#4\n\t"

/* x0 = a_ptr1 (top) */
/* x1 = a_ptr2 */
/* x2 = a_ptr3 */
/* x3 = a_ptr4 (or pref_head when M == 3) */
/* x4 = b_ptr */
/* w5 = k_left */
/* x8 - x11 for pref head */
/* x12 - x15 for c_tmp1 - c_tmp4 */

/* macro for GEMM with packing pattern NO.#3 */
/* mdim = 3, 4; ndim = 5 - 8 */
#define FUNC_PACK3(mdim, ndim) \
static inline void sgemm_skinny1_a35_m##mdim##n##ndim(\
  const float * __restrict__ a_ptr, const float * __restrict__ b_scr,\
  float * __restrict__ c_ptr, uint32_t K, uint32_t LDA, uint32_t LDC,\
  uint8_t c_rowmajor, const float * __restrict__ beta_addr) {\
  __asm__ __volatile__(\
    "mov x4,%[b_scr]\n\t"\
    "mov x0,%[a_ptr]; add x1,%[a_ptr],%w[LDA],UXTW #2\n\t"\
    "add x2,%[a_ptr],%w[LDA],UXTW #3; add x3,x1,%w[LDA],UXTW #3\n\t"\
    "add x8,x0,%w[LDA],UXTW #4; add x9,x1,%w[LDA],UXTW #4\n\t"\
    "add x10,x2,%w[LDA],UXTW #4; add x11,x3,%w[LDA],UXTW #4\n\t"\
    "mov w5,%w[K]\n\t"\
    INIT_M##mdim##N##ndim\
    "cmp w5,#2; b.lt 4f\n\t"\
    KERNEL_M##mdim##N##ndim##_PRELOAD2\
    "cmp w5,#10; b.lt 7f\n\t"\
    ".balign 16; 8:\n\t"\
    KERNEL_M##mdim##N##ndim##_MAIN8 "b.ge 8b\n\t"\
    "7:\n\t"\
    "cmp w5,#6; b.lt 1f\n\t"\
    KERNEL_M##mdim##N##ndim##_MAIN4\
    "1:\n\t"\
    "cmp w5,#4; b.lt 2f\n\t"\
    KERNEL_M##mdim##N##ndim##_TAIL4 "b 4f\n\t"\
    "2:\n\t"\
    KERNEL_M##mdim##N##ndim##_TAIL2\
    "4:\n\t"\
    "cmp w5,#1; b.lt 5f\n\t"\
    KERNEL_M##mdim##N##ndim##_FIN1\
    "5:\n\t"\
    "cmp %w[c_rowmajor],#0; b.eq 6f\n\t"\
    INIT_SAVE_M##mdim##_CR SAVE_M##mdim##N##ndim(CR) "b 7f\n\t"\
    "6:\n\t"\
    INIT_SAVE_CC SAVE_M##mdim##N##ndim(CC)\
    "7:\n\t"\
   ::[a_ptr]"r"(a_ptr), [b_scr]"r"(b_scr), [c_ptr]"r"(c_ptr),\
     [LDA]"r"(LDA), [LDC]"r"(LDC), [K]"r"(K),\
     [beta_addr]"r"(beta_addr), [c_rowmajor]"r"(c_rowmajor)\
   :"cc","memory","x0","x1","x2","x3","x4","x5",\
    "x8","x9","x10","x11","x12","x13","x14","x15",\
    "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",\
    "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21",\
    "v22","v23","v24","v25","v26","v27","v28","v29","v30","v31");\
}

/* acc layout for m4n4 kernel */
/* m0n0 v16 v17 v18 v19 m0n4 */
/* m1n0 v20 v21 v22 v23 m1n4 */
/* m2n0 v24 v25 v26 v27 m2n4 */
/* m3n0 v28 v29 v30 v31 m3n4 */
/* b-holder layout for m4n4 kernel */
/* n0 v4 v5 v6 v7 */
/* a-holder layout for m4n4 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2, a_ptr4->v3 */

#define INIT_M4N4 \
  INIT_4V(16, 20, 24, 28) INIT_4V(17, 21, 25, 29)\
  INIT_4V(18, 22, 26, 30) INIT_4V(19, 23, 27, 31)

#define SAVE_M4N4(mode) \
  UNIT_SAVE_M4N1_##mode(16, 20, 24, 28) UNIT_SAVE_M4N1_##mode(17, 21, 25, 29)\
  UNIT_SAVE_M4N1_##mode(18, 22, 26, 30) UNIT_SAVE_M4N1_##mode(19, 23, 27, 31)

#define KERNEL_M4N4_PRELOAD2 \
  "ldr d0,[x0],#8\n\t"\
  "ldr d4,[x4]; ldr d5,[x4,#8]; ldr d6,[x4,#16]; add x4,x4,#32\n\t"

#define KERNEL_M4N4_MAIN8 \
  "ldr d1,[x1],#8\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "ldr d4,[x4]\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; ldr d6,[x4,#16]\n\t"\
  "prfm pldl1keep,[x0,#64]; fmla v31.2s,v3.2s,v7.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "ldr d4,[x4,#32]\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#40]; ldr d6,[x4,#48]\n\t"\
  "prfm pldl1keep,[x1,#64]; fmla v31.2s,v3.2s,v7.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#56]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "ldr d4,[x4,#64]\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#72]; ldr d6,[x4,#80]\n\t"\
  "prfm pldl1keep,[x2,#64]; fmla v31.2s,v3.2s,v7.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#88]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "ldr d4,[x4,#96]\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "ldr d0,[x0],#8; sub w5,w5,#8\n\t"\
  "ldr d5,[x4,#104]; ldr d6,[x4,#112]; cmp w5,#10\n\t"\
  "prfm pldl1keep,[x3,#64]; fmla v31.2s,v3.2s,v7.2s; add x4,x4,#128\n\t"

#define KERNEL_M4N4_MAIN4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "ldr d4,[x4]\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; ldr d6,[x4,#16]\n\t"\
  "sub w5,w5,#4; fmla v31.2s,v3.2s,v7.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "ldr d4,[x4,#32]\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#40]; ldr d6,[x4,#48]\n\t"\
  "add x4,x4,#64; fmla v31.2s,v3.2s,v7.2s\n\t"

#define KERNEL_M4N4_TAIL4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "ldr d4,[x4]\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; ldr d6,[x4,#16]\n\t"\
  "prfm pldl1keep,[x8]; sub w5,w5,#4\n\t"\
  "prfm pldl1keep,[x9]; fmla v31.2s,v3.2s,v7.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "prfm pldl1keep,[x10]\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "add x4,x4,#32\n\t"\
  "prfm pldl1keep,[x11]; fmla v31.2s,v3.2s,v7.2s\n\t"

#define KERNEL_M4N4_TAIL2 \
  "ldr d1,[x1],#8\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "prfm pldl1keep,[x9]\n\t"\
  "prfm pldl1keep,[x10]; sub w5,w5,#2\n\t"\
  "prfm pldl1keep,[x11]; fmla v31.2s,v3.2s,v7.2s\n\t"

#define KERNEL_M4N4_FIN1 \
  "ldr s0,[x0],#4; ldr s4,[x4]; ldr s5,[x4,#4]; ldr s6,[x4,#8]\n\t"\
  "ldr s1,[x1],#4\n\t" FMA_3V(16, 17, 18, 0, 0, 0, 4, 5, 6)\
  "ldr s2,[x2],#4\n\t" FMA_3V(20, 21, 22, 1, 1, 1, 4, 5, 6)\
  "ldr s3,[x3],#4\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 4, 5, 6)\
  "ldr s7,[x4,#12]\n\t" FMA_3V(28, 29, 30, 3, 3, 3, 4, 5, 6)\
  "add x4,x4,#16\n\t" FMA_3V(19, 23, 27, 0, 1, 2, 7, 7, 7)\
  "fmla v31.2s,v3.2s,v7.2s\n\t"


/* acc layout for m4n5 kernel */
/* m0n0 v12 v13 v14 v15 v16 m0n5 */
/* m1n0 v17 v18 v19 v20 v21 m1n5 */
/* m2n0 v22 v23 v24 v25 v26 m2n5 */
/* m3n0 v27 v28 v29 v30 v31 m3n5 */
/* b-holder layout for m4n5 kernel */
/* n0 v4 v5 v6 v7 v8 */
/* a-holder layout for m4n5 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2, a_ptr4->v3 */

#define INIT_M4N5 \
  INIT_4V(12, 17, 22, 27) INIT_4V(13, 18, 23, 28) INIT_4V(14, 19, 24, 29)\
  INIT_4V(15, 20, 25, 30) INIT_4V(16, 21, 26, 31)

#define SAVE_M4N5(mode) \
  UNIT_SAVE_M4N1_##mode(12, 17, 22, 27) UNIT_SAVE_M4N1_##mode(13, 18, 23, 28)\
  UNIT_SAVE_M4N1_##mode(14, 19, 24, 29) UNIT_SAVE_M4N1_##mode(15, 20, 25, 30)\
  UNIT_SAVE_M4N1_##mode(16, 21, 26, 31)

#define KERNEL_M4N5_PRELOAD2 \
  "ldr d0,[x0],#8\n\t"\
  "ldr d4,[x4]; ldr d5,[x4,#8]; ldr d6,[x4,#16]; add x4,x4,#40\n\t"

#define KERNEL_M4N5_MAIN8 \
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-16]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#-8]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; ldr d6,[x4,#16]\n\t"\
  "prfm pldl1keep,[x0,#64]; fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#32]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4,#40]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "ldr d0,[x0],#8; ldr d5,[x4,#48]; ldr d6,[x4,#56]; sub w5,w5,#8\n\t"\
  "prfm pldl1keep,[x1,#64]; fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#64]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#72]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4,#80]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "ldr d0,[x0],#8; ldr d5,[x4,#88]; ldr d6,[x4,#96]; cmp w5,#10\n\t"\
  "prfm pldl1keep,[x2,#64]; fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#104]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#112]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4,#120]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "ldr d0,[x0],#8; ldr d5,[x4,#128]; ldr d6,[x4,#136]; add x4,x4,#160\n\t"\
  "prfm pldl1keep,[x3,#64]; fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"

#define KERNEL_M4N5_MAIN4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-16]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#-8]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; ldr d6,[x4,#16]; sub w5,w5,#4\n\t"\
  "fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#32]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4,#40]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "ldr d0,[x0],#8; ldr d5,[x4,#48]; ldr d6,[x4,#56]; add x4,x4,#80\n\t"\
  "fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"

#define KERNEL_M4N5_TAIL4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-16]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#-8]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; ldr d6,[x4,#16]\n\t"\
  "prfm pldl1keep,[x8]\n\t"\
  "prfm pldl1keep,[x9]; fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#32]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "prfm pldl1keep,[x10]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "add x4,x4,#40; sub w5,w5,#4\n\t"\
  "prfm pldl1keep,[x11]; fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"

#define KERNEL_M4N5_TAIL2 \
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-16]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#-8]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "prfm pldl1keep,[x9]; prfm pldl1keep,[x10]; prfm pldl1keep,[x11]\n\t"\
  "sub w5,w5,#2; fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"

#define KERNEL_M4N5_FIN1 \
  "ldr s0,[x0],#4; ldr s4,[x4]; ldr s5,[x4,#4]; ldr s6,[x4,#8]\n\t"\
  "ldr s1,[x1],#4\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr s2,[x2],#4\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr s3,[x3],#4\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr s7,[x4,#12]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr s8,[x4,#16]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "add x4,x4,#20\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"


/* acc layout for m4n6 kernel */
/* m0n0 v8 v9 v10 v11 v12 v13 m0n6 */
/* m1n0 v14 v15 v16 v17 v18 v19 m1n6 */
/* m2n0 v20 v21 v22 v23 v24 v25 m2n6 */
/* m3n0 v26 v27 v28 v29 v30 v31 m3n6 */
/* b-holder layout for m4n6 kernel */
/* n0 v4 v5 v6 v7 */
/* a-holder layout for m4n5 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2, a_ptr4->v3 */

#define INIT_M4N6 \
  INIT_4V(8, 14, 20, 26) INIT_4V(9, 15, 21, 27) INIT_4V(10, 16, 22, 28)\
  INIT_4V(11, 17, 23, 29) INIT_4V(12, 18, 24, 30) INIT_4V(13, 19, 25, 31)

#define SAVE_M4N6(mode) \
  UNIT_SAVE_M4N1_##mode(8, 14, 20, 26) UNIT_SAVE_M4N1_##mode(9, 15, 21, 27)\
  UNIT_SAVE_M4N1_##mode(10, 16, 22, 28) UNIT_SAVE_M4N1_##mode(11, 17, 23, 29)\
  UNIT_SAVE_M4N1_##mode(12, 18, 24, 30) UNIT_SAVE_M4N1_##mode(13, 19, 25, 31)

#define KERNEL_M4N6_PRELOAD2 \
  "ldr d0,[x0],#8\n\t"\
  "ldr d4,[x4]; ldr d5,[x4,#8]; ldr d6,[x4,#16]; add x4,x4,#48\n\t"

#define KERNEL_M4N6_MAIN8 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-16]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-8]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; ldr d7,[x4,#16]; sub w5,w5,#8\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d5,[x4,#32]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d7,[x4,#40]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#48]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 6, 5, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#56]; ldr d6,[x4,#64]; cmp w5,#10\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 7, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#72]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#80]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#88]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#96]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ldr d0,[x0],#8; ldr d5,[x4,#104]; ldr d7,[x4,#112]\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#120]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d5,[x4,#128]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d7,[x4,#136]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#144]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 6, 5, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#152]; ldr d6,[x4,#160]; add x4,x4,#192\n\t"\
  "prfm pldl1keep,[x3,#64]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 7, 7, 7)

#define KERNEL_M4N6_MAIN4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-16]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-8]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; ldr d7,[x4,#16]; sub w5,w5,#4\n\t"\
  FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d5,[x4,#32]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d7,[x4,#40]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#48]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 6, 5, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#56]; ldr d6,[x4,#64]; add x4,x4,#96\n\t"\
  FMA_3V(19, 25, 31, 1, 2, 3, 7, 7, 7)

#define KERNEL_M4N6_TAIL4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-16]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-8]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; ldr d7,[x4,#16]; sub w5,w5,#4\n\t"\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d5,[x4,#32]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d7,[x4,#40]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 6, 5, 7)\
  "prfm pldl1keep,[x10]; add x4,x4,#48\n\t"\
  "prfm pldl1keep,[x11]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 7, 7, 7)

#define KERNEL_M4N6_TAIL2 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-16]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-8]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "prfm pldl1keep,[x9]; prfm pldl1keep,[x10]; sub w5,w5,#2\n\t"\
  "prfm pldl1keep,[x11]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)

#define KERNEL_M4N6_FIN1 \
  "ldr s0,[x0],#4; ldr s4,[x4]; ldr s5,[x4,#4]; ldr s6,[x4,#8]\n\t"\
  "ldr s1,[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr s2,[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr s3,[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr s7,[x4,#12]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr s5,[x4,#16]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr s6,[x4,#20]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "add x4,x4,#24\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)


/* acc layout for m3n7 kernel */
/* m1n0 v11 v12 v13 v14 v15 v16 v17 m0n7 */
/* m2n0 v18 v19 v20 v21 v22 v23 v24 m1n7 */
/* m3n0 v25 v26 v27 v28 v29 v30 v31 m2n7 */
/* b-holder layout for m3n7 kernel */
/* n0 v3 v4 v5 v6 v7 v8 v9 */
/* a-holder layout for m3n7 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2 */

#define INIT_M3N7 \
  INIT_3V(11, 18, 25) INIT_3V(12, 19, 26) INIT_3V(13, 20, 27)\
  INIT_3V(14, 21, 28) INIT_3V(15, 22, 29) INIT_3V(16, 23, 30)\
  INIT_3V(17, 24, 31)

#define SAVE_M3N7(mode) \
  UNIT_SAVE_M3N1_##mode(11, 18, 25) UNIT_SAVE_M3N1_##mode(12, 19, 26)\
  UNIT_SAVE_M3N1_##mode(13, 20, 27) UNIT_SAVE_M3N1_##mode(14, 21, 28)\
  UNIT_SAVE_M3N1_##mode(15, 22, 29) UNIT_SAVE_M3N1_##mode(16, 23, 30)\
  UNIT_SAVE_M3N1_##mode(17, 24, 31)

#define KERNEL_M3N7_PRELOAD2 \
  "ldr d0,[x0],#8\n\t"\
  "ldr d3,[x4]; ldr d4,[x4,#8]; ldr d5,[x4,#16]; add x4,x4,#56\n\t"

#define KERNEL_M3N7_MAIN8 \
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-16]; ldr d9,[x4,#-8]; ldr d3,[x4]\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#40]; ldr d9,[x4,#48]; ldr d3,[x4,#56]; sub w5,w5,#8\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t"\
  "ldr d4,[x4,#64]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#72]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#80]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#88]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#96]; ldr d9,[x4,#104]; ldr d3,[x4,#112]; cmp w5,#10\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t"\
  "ldr d4,[x4,#120]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#128]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#136]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#144]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#152]; ldr d9,[x4,#160]; ldr d3,[x4,#168]; add x4,x4,#224\n\t"\
  "prfm pldl1keep,[x3,#64]\n\t"\
  "ldr d4,[x4,#-48]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#-40]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)

#define KERNEL_M3N7_MAIN4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-16]; ldr d9,[x4,#-8]; ldr d3,[x4]; sub w5,w5,#4\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#40]; ldr d9,[x4,#48]; ldr d3,[x4,#56]; add x4,x4,#112\n\t"\
  "ldr d4,[x4,#-48]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#-40]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)

#define KERNEL_M3N7_TAIL4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-16]; ldr d9,[x4,#-8]; ldr d3,[x4]; sub w5,w5,#4\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#40]; ldr d9,[x4,#48]; add x4,x4,#56\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)

#define KERNEL_M3N7_TAIL2 \
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-16]; ldr d9,[x4,#-8]; sub w5,w5,#2\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)

#define KERNEL_M3N7_FIN1 \
  "ldr s0,[x0],#4; ldr s3,[x4]; ldr s4,[x4,#4]; ldr s5,[x4,#8]\n\t"\
  "ldr s1,[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr s2,[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr s6,[x4,#12]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr s7,[x4,#16]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr s8,[x4,#20]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr s9,[x4,#24]\n\t" FMA_3V(16, 23, 30, 0, 1, 2, 8, 8, 8)\
  "add x4,x4,#28\n\t" FMA_3V(17, 24, 31, 0, 1, 2, 9, 9, 9)


/* acc layout for m3n8 kernel */
/* m1n0 v8 v9 v10 v11 v12 v13 v14 v15 m0n8 */
/* m2n0 v16 v17 v18 v19 v20 v21 v22 v23 m1n8 */
/* m3n0 v24 v25 v26 v27 v28 v29 v30 v31 m2n8 */
/* b-holder layout for m3n8 kernel */
/* n0 v3 v4 v5 v6 v7 */
/* a-holder layout for m3n8 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2 */

#define INIT_M3N8 \
  INIT_3V(8, 16, 24) INIT_3V(9, 17, 25) INIT_3V(10, 18, 26)\
  INIT_3V(11, 19, 27) INIT_3V(12, 20, 28) INIT_3V(13, 21, 29)\
  INIT_3V(14, 22, 30) INIT_3V(15, 23, 31)

#define SAVE_M3N8(mode) \
  UNIT_SAVE_M3N1_##mode(8, 16, 24) UNIT_SAVE_M3N1_##mode(9, 17, 25)\
  UNIT_SAVE_M3N1_##mode(10, 18, 26) UNIT_SAVE_M3N1_##mode(11, 19, 27)\
  UNIT_SAVE_M3N1_##mode(12, 20, 28) UNIT_SAVE_M3N1_##mode(13, 21, 29)\
  UNIT_SAVE_M3N1_##mode(14, 22, 30) UNIT_SAVE_M3N1_##mode(15, 23, 31)

#define KERNEL_M3N8_PRELOAD2 \
  "ldr d0,[x0],#8\n\t"\
  "ldr d3,[x4]; ldr d4,[x4,#8]; ldr d5,[x4,#16]; add x4,x4,#64\n\t"

#define KERNEL_M3N8_MAIN8 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-40]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; ldr d3,[x4]\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#48]; ldr d7,[x4,#56]; ldr d3,[x4,#64]; sub w5,w5,#8\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t"\
  "ldr d4,[x4,#72]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#80]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#88]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#96]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#104]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#112]; ldr d7,[x4,#120]; ldr d3,[x4,#128]; cmp w5,#10\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t"\
  "ldr d4,[x4,#136]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#144]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#152]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#160]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#168]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#176]; ldr d7,[x4,#184]; ldr d3,[x4,#192]; add x4,x4,#256\n\t"\
  "prfm pldl1keep,[x3,#64]\n\t"\
  "ldr d4,[x4,#-56]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#-48]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)

#define KERNEL_M3N8_MAIN4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-40]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; ldr d3,[x4]; sub w5,w5,#4\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#48]; ldr d7,[x4,#56]; ldr d3,[x4,#64]; add x4,x4,#128\n\t"\
  "ldr d4,[x4,#-56]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#-48]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)

#define KERNEL_M3N8_TAIL4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-40]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; ldr d3,[x4]; sub w5,w5,#4\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#48]; ldr d7,[x4,#56]; add x4,x4,#64\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)

#define KERNEL_M3N8_TAIL2 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-40]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; sub w5,w5,#2\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)

#define KERNEL_M3N8_FIN1 \
  "ldr s0,[x0],#4; ldr s3,[x4]; ldr s4,[x4,#4]; ldr s5,[x4,#8]\n\t"\
  "ldr s1,[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr s2,[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr s6,[x4,#12]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr s7,[x4,#16]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr s5,[x4,#20]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr s6,[x4,#24]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr s7,[x4,#28]\n\t" FMA_3V(14, 22, 30, 0, 1, 2, 6, 6, 6)\
  "add x4,x4,#32\n\t" FMA_3V(15, 23, 31, 0, 1, 2, 7, 7, 7)

FUNC_PACK3(4, 4)

FUNC_PACK3(4, 5)

FUNC_PACK3(4, 6)

FUNC_PACK3(3, 7)

FUNC_PACK3(3, 8)

/* macro for GEMM with packing pattern NO.#0 */
/* mdim = 3, 4; ndim = 10, 12, 14, 16 */
#define FUNC_PACK0(mdim, ndim) \
static inline void sgemm_skinny1_a35_m##mdim##n##ndim(\
  const float * __restrict__ a_ptr, const float * __restrict__ b_scr,\
  float * __restrict__ c_ptr, uint32_t K, uint32_t LDA, uint32_t LDC,\
  uint8_t c_rowmajor, const float * __restrict__ beta_addr) {\
  __asm__ __volatile__(\
    "mov x4,%[b_scr]\n\t"\
    "mov x0,%[a_ptr]; add x1,%[a_ptr],%w[LDA],UXTW #2\n\t"\
    "add x2,%[a_ptr],%w[LDA],UXTW #3; add x3,x1,%w[LDA],UXTW #3\n\t"\
    "add x8,x0,%w[LDA],UXTW #4; add x9,x1,%w[LDA],UXTW #4\n\t"\
    "add x10,x2,%w[LDA],UXTW #4; add x11,x3,%w[LDA],UXTW #4\n\t"\
    "mov w5,%w[K]\n\t"\
    INIT_M##mdim##N##ndim\
    "cmp w5,#1; b.lt 4f\n\t"\
    KERNEL_M##mdim##N##ndim##_PRELOAD1\
    "cmp w5,#5; b.lt 1f\n\t"\
    ".balign 16; 8:\n\t"\
    KERNEL_M##mdim##N##ndim##_MAIN4 "b.ge 8b\n\t"\
    "1:\n\t"\
    "cmp w5,#3; b.lt 2f\n\t"\
    KERNEL_M##mdim##N##ndim##_MAIN2\
    "2:\n\t"\
    "cmp w5,#2; b.ne 3f\n\t"\
    KERNEL_M##mdim##N##ndim##_TAIL2 "b 4f\n\t"\
    "3:\n\t"\
    KERNEL_M##mdim##N##ndim##_TAIL1\
    "4:\n\t"\
    "cmp %w[c_rowmajor],#0; b.eq 6f\n\t"\
    INIT_SAVE_M##mdim##_CR SAVE_M##mdim##N##ndim(CR) "b 7f\n\t"\
    "6:\n\t"\
    INIT_SAVE_CC SAVE_M##mdim##N##ndim(CC)\
    "7:\n\t"\
   ::[a_ptr]"r"(a_ptr), [b_scr]"r"(b_scr), [c_ptr]"r"(c_ptr),\
     [LDA]"r"(LDA), [LDC]"r"(LDC), [K]"r"(K),\
     [beta_addr]"r"(beta_addr), [c_rowmajor]"r"(c_rowmajor)\
   :"cc","memory","x0","x1","x2","x3","x4","x5",\
    "x8","x9","x10","x11","x12","x13","x14","x15",\
    "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",\
    "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21",\
    "v22","v23","v24","v25","v26","v27","v28","v29","v30","v31");\
}

/* acc layout for m4n10 kernel */
/* m0n0 v10 v11 v12 v13 v14 m0n10 */
/* m1n0 v15 v16 v17 v18 v19 m1n10 */
/* m2n0 v20 v21 v22 v23 v24 m2n10 */
/* m3n0 v25 v26 v27 v28 v29 m3n10 */
/* b-holder layout for m4n10 kernel */
/* n0 v5 v6 v7 v8 v9 n10 */
/* a-holder layout for m4n10 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2, a_ptr4->v3 */

#define INIT_M4N10 \
  INIT_4V(10, 15, 20, 25) INIT_4V(11, 16, 21, 26)\
  INIT_4V(12, 17, 22, 27) INIT_4V(13, 18, 23, 28)\
  INIT_4V(14, 19, 24, 29)

#define SAVE_M4N10(mode) \
  UNIT_SAVE_M4N2_##mode(10, 15, 20, 25) UNIT_SAVE_M4N2_##mode(11, 16, 21, 26)\
  UNIT_SAVE_M4N2_##mode(12, 17, 22, 27) UNIT_SAVE_M4N2_##mode(13, 18, 23, 28)\
  UNIT_SAVE_M4N2_##mode(14, 19, 24, 29)

#define KERNEL_M4N10_PRELOAD1 \
  "ld1r {v0.2s},[x0],#4\n\t"\
  "ldr d5,[x4]; ldr d6,[x4,#8]; ldr d7,[x4,#16]; add x4,x4,#40\n\t"

#define KERNEL_M4N10_MAIN4 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(10, 11, 12, 0, 0, 0, 5, 6, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(15, 16, 17, 1, 1, 1, 5, 6, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 5, 6, 7)\
  "ldr d8,[x4,#-16]\n\t" FMA_3V(25, 26, 27, 3, 3, 3, 5, 6, 7)\
  "ldr d9,[x4,#-8]\n\t" FMA_3V(13, 18, 23, 0, 1, 2, 8, 8, 8)\
  "ldr d5,[x4]\n\t" FMA_3V(14, 19, 24, 0, 1, 2, 9, 9, 9)\
  "ld1r {v0.2s},[x0],#4; ldr d6,[x4,#8]; ldr d7,[x4,#16]\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t"\
  "fmla v28.2s,v3.2s,v8.2s; fmla v29.2s,v3.2s,v9.2s\n\t"\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(10, 11, 12, 0, 0, 0, 5, 6, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(15, 16, 17, 1, 1, 1, 5, 6, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 5, 6, 7)\
  "ldr d8,[x4,#24]\n\t" FMA_3V(25, 26, 27, 3, 3, 3, 5, 6, 7)\
  "ldr d9,[x4,#32]\n\t" FMA_3V(13, 18, 23, 0, 1, 2, 8, 8, 8)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(14, 19, 24, 0, 1, 2, 9, 9, 9)\
  "ld1r {v0.2s},[x0],#4; ldr d6,[x4,#48]; ldr d7,[x4,#56]\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t"\
  "fmla v28.2s,v3.2s,v8.2s; fmla v29.2s,v3.2s,v9.2s\n\t"\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(10, 11, 12, 0, 0, 0, 5, 6, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(15, 16, 17, 1, 1, 1, 5, 6, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 5, 6, 7)\
  "ldr d8,[x4,#64]\n\t" FMA_3V(25, 26, 27, 3, 3, 3, 5, 6, 7)\
  "ldr d9,[x4,#72]\n\t" FMA_3V(13, 18, 23, 0, 1, 2, 8, 8, 8)\
  "ldr d5,[x4,#80]\n\t" FMA_3V(14, 19, 24, 0, 1, 2, 9, 9, 9)\
  "ld1r {v0.2s},[x0],#4; ldr d6,[x4,#88]; ldr d7,[x4,#96]\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t"\
  "fmla v28.2s,v3.2s,v8.2s; fmla v29.2s,v3.2s,v9.2s\n\t"\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(10, 11, 12, 0, 0, 0, 5, 6, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(15, 16, 17, 1, 1, 1, 5, 6, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 5, 6, 7)\
  "ldr d8,[x4,#104]\n\t" FMA_3V(25, 26, 27, 3, 3, 3, 5, 6, 7)\
  "ldr d9,[x4,#112]\n\t" FMA_3V(13, 18, 23, 0, 1, 2, 8, 8, 8)\
  "ldr d5,[x4,#120]\n\t" FMA_3V(14, 19, 24, 0, 1, 2, 9, 9, 9)\
  "ld1r {v0.2s},[x0],#4; ldr d6,[x4,#128]; ldr d7,[x4,#136]; add x4,x4,#160\n\t"\
  "prfm pldl1keep,[x3,#64]; sub w5,w5,#4\n\t"\
  "fmla v28.2s,v3.2s,v8.2s; cmp w5,#5; fmla v29.2s,v3.2s,v9.2s\n\t"

#define KERNEL_M4N10_MAIN2 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(10, 11, 12, 0, 0, 0, 5, 6, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(15, 16, 17, 1, 1, 1, 5, 6, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 5, 6, 7)\
  "ldr d8,[x4,#-16]\n\t" FMA_3V(25, 26, 27, 3, 3, 3, 5, 6, 7)\
  "ldr d9,[x4,#-8]\n\t" FMA_3V(13, 18, 23, 0, 1, 2, 8, 8, 8)\
  "ldr d5,[x4]\n\t" FMA_3V(14, 19, 24, 0, 1, 2, 9, 9, 9)\
  "ld1r {v0.2s},[x0],#4; ldr d6,[x4,#8]; ldr d7,[x4,#16]\n\t"\
  "fmla v28.2s,v3.2s,v8.2s; fmla v29.2s,v3.2s,v9.2s\n\t"\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(10, 11, 12, 0, 0, 0, 5, 6, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(15, 16, 17, 1, 1, 1, 5, 6, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 5, 6, 7)\
  "ldr d8,[x4,#24]\n\t" FMA_3V(25, 26, 27, 3, 3, 3, 5, 6, 7)\
  "ldr d9,[x4,#32]\n\t" FMA_3V(13, 18, 23, 0, 1, 2, 8, 8, 8)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(14, 19, 24, 0, 1, 2, 9, 9, 9)\
  "ld1r {v0.2s},[x0],#4; ldr d6,[x4,#48]; ldr d7,[x4,#56]; add x4,x4,#80\n\t"\
  "sub w5,w5,#2\n\t"\
  "fmla v28.2s,v3.2s,v8.2s; fmla v29.2s,v3.2s,v9.2s\n\t"

#define KERNEL_M4N10_TAIL2 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(10, 11, 12, 0, 0, 0, 5, 6, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(15, 16, 17, 1, 1, 1, 5, 6, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 5, 6, 7)\
  "ldr d8,[x4,#-16]\n\t" FMA_3V(25, 26, 27, 3, 3, 3, 5, 6, 7)\
  "ldr d9,[x4,#-8]\n\t" FMA_3V(13, 18, 23, 0, 1, 2, 8, 8, 8)\
  "ldr d5,[x4]\n\t" FMA_3V(14, 19, 24, 0, 1, 2, 9, 9, 9)\
  "ld1r {v0.2s},[x0],#4; ldr d6,[x4,#8]; ldr d7,[x4,#16]\n\t"\
  "prfm pldl1keep,[x8]; prfm pldl1keep,[x9]\n\t"\
  "fmla v28.2s,v3.2s,v8.2s; fmla v29.2s,v3.2s,v9.2s\n\t"\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(10, 11, 12, 0, 0, 0, 5, 6, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(15, 16, 17, 1, 1, 1, 5, 6, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 5, 6, 7)\
  "ldr d8,[x4,#24]\n\t" FMA_3V(25, 26, 27, 3, 3, 3, 5, 6, 7)\
  "ldr d9,[x4,#32]\n\t" FMA_3V(13, 18, 23, 0, 1, 2, 8, 8, 8)\
  "prfm pldl1keep,[x10]; sub w5,w5,#2\n\t" FMA_3V(14, 19, 24, 0, 1, 2, 9, 9, 9)\
  "prfm pldl1keep,[x11]; add x4,x4,#40\n\t"\
  "fmla v28.2s,v3.2s,v8.2s; fmla v29.2s,v3.2s,v9.2s\n\t"

#define KERNEL_M4N10_TAIL1 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(10, 11, 12, 0, 0, 0, 5, 6, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(15, 16, 17, 1, 1, 1, 5, 6, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 5, 6, 7)\
  "ldr d8,[x4,#-16]\n\t" FMA_3V(25, 26, 27, 3, 3, 3, 5, 6, 7)\
  "ldr d9,[x4,#-8]\n\t" FMA_3V(13, 18, 23, 0, 1, 2, 8, 8, 8)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(14, 19, 24, 0, 1, 2, 9, 9, 9)\
  "prfm pldl1keep,[x9]; prfm pldl1keep,[x10]; prfm pldl1keep,[x11]\n\t"\
  "sub w5,w5,#1\n\t"\
  "fmla v28.2s,v3.2s,v8.2s; fmla v29.2s,v3.2s,v9.2s\n\t"


/* acc layout for m4n12 kernel */
/* m0n0 v8 v9 v10 v11 v12 v13 m0n12 */
/* m1n0 v14 v15 v16 v17 v18 v19 m1n12 */
/* m2n0 v20 v21 v22 v23 v24 v25 m2n12 */
/* m3n0 v26 v27 v28 v29 v30 v31 m3n12 */
/* b-holder layout for m4n12 kernel */
/* n0 v4 v5 v6/v7 v7/v6 v5 v6/v7 n12 */
/* a-holder layout for m4n12 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2, a_ptr4->v3 */

#define INIT_M4N12 \
  INIT_4V(8, 14, 20, 26) INIT_4V(9, 15, 21, 27)\
  INIT_4V(10, 16, 22, 28) INIT_4V(11, 17, 23, 29)\
  INIT_4V(12, 18, 24, 30) INIT_4V(13, 19, 25, 31)

#define SAVE_M4N12(mode) \
  UNIT_SAVE_M4N2_##mode(8, 14, 20, 26) UNIT_SAVE_M4N2_##mode(9, 15, 21, 27)\
  UNIT_SAVE_M4N2_##mode(10, 16, 22, 28) UNIT_SAVE_M4N2_##mode(11, 17, 23, 29)\
  UNIT_SAVE_M4N2_##mode(12, 18, 24, 30) UNIT_SAVE_M4N2_##mode(13, 19, 25, 31)

#define KERNEL_M4N12_PRELOAD1 \
  "ld1r {v0.2s},[x0],#4\n\t"\
  "ldr d4,[x4]; ldr d5,[x4,#8]; ldr d6,[x4,#16]; add x4,x4,#48\n\t"

#define KERNEL_M4N12_MAIN4 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-16]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-8]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ld1r {v0.2s},[x0],#4; ldr d5,[x4,#8]; ldr d7,[x4,#16]\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d5,[x4,#32]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d7,[x4,#40]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#48]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 6, 5, 7)\
  "ld1r {v0.2s},[x0],#4; ldr d5,[x4,#56]; ldr d6,[x4,#64]\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 7, 7, 7)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#72]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#80]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#88]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#96]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ld1r {v0.2s},[x0],#4; ldr d5,[x4,#104]; ldr d7,[x4,#112]\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#120]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d5,[x4,#128]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d7,[x4,#136]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#144]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 6, 5, 7)\
  "ld1r {v0.2s},[x0],#4; sub w5,w5,#4; ldr d5,[x4,#152]; cmp w5,#5\n\t"\
  "ldr d6,[x4,#160]; add x4,x4,#192\n\t"\
  "prfm pldl1keep,[x3,#64]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 7, 7, 7)

#define KERNEL_M4N12_MAIN2 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-16]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-8]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ld1r {v0.2s},[x0],#4; ldr d5,[x4,#8]; ldr d7,[x4,#16]\n\t"\
  FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d5,[x4,#32]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d7,[x4,#40]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#48]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 6, 5, 7)\
  "ld1r {v0.2s},[x0],#4; ldr d5,[x4,#56]; ldr d6,[x4,#64]; add x4,x4,#96\n\t"\
  "sub w5,w5,#2\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 7, 7, 7)

#define KERNEL_M4N12_TAIL2 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-16]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-8]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ld1r {v0.2s},[x0],#4; ldr d5,[x4,#8]; ldr d7,[x4,#16]\n\t"\
  "prfm pldl1keep,[x8]; prfm pldl1keep,[x9]\n\t"\
  FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d5,[x4,#32]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d7,[x4,#40]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "prfm pldl1keep,[x10]; add x4,x4,#48\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 6, 5, 7)\
  "prfm pldl1keep,[x11]; sub w5,w5,#2\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 7, 7, 7)

#define KERNEL_M4N12_TAIL1 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-16]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-8]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "prfm pldl1keep,[x8]; prfm pldl1keep,[x9]; prfm pldl1keep,[x10]\n\t"\
  "sub w5,w5,#1\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "prfm pldl1keep,[x11]\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)


/* acc layout for m3n14 kernel */
/* m0n0 v11 v12 v13 v14 v15 v16 v17 m0n14 */
/* m1n0 v18 v19 v20 v21 v22 v23 v24 m1n14 */
/* m2n0 v25 v26 v27 v28 v29 v30 v31 m2n14 */
/* b-holder layout for m3n14 kernel */
/* n0 v3 v4 v5 v6 v7 v8 v9 n14 */
/* a-holder layout for m3n14 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2 */

#define INIT_M3N14 \
  INIT_3V(11, 18, 25) INIT_3V(12, 19, 26) INIT_3V(13, 20, 27)\
  INIT_3V(14, 21, 28) INIT_3V(15, 22, 29) INIT_3V(16, 23, 30)\
  INIT_3V(17, 24, 31)

#define SAVE_M3N14(mode) \
  UNIT_SAVE_M3N2_##mode(11, 18, 25) UNIT_SAVE_M3N2_##mode(12, 19, 26)\
  UNIT_SAVE_M3N2_##mode(13, 20, 27) UNIT_SAVE_M3N2_##mode(14, 21, 28)\
  UNIT_SAVE_M3N2_##mode(15, 22, 29) UNIT_SAVE_M3N2_##mode(16, 23, 30)\
  UNIT_SAVE_M3N2_##mode(17, 24, 31)

#define KERNEL_M3N14_PRELOAD1 \
  "ld1r {v0.2s},[x0],#4\n\t"\
  "ldr d3,[x4]; ldr d4,[x4,#8]; ldr d5,[x4,#16]; add x4,x4,#56\n\t"

#define KERNEL_M3N14_MAIN4 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-16]; ldr d9,[x4,#-8]; ldr d3,[x4]; ldr d4,[x4,#8]\n\t"\
  FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(16, 17, 23, 0, 0, 1, 8, 9, 8)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(24, 30, 31, 1, 2, 2, 9, 8, 9)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#40]; ldr d9,[x4,#48]; ldr d3,[x4,#56]; ldr d4,[x4,#64]\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#72]\n\t" FMA_3V(16, 17, 23, 0, 0, 1, 8, 9, 8)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(24, 30, 31, 1, 2, 2, 9, 8, 9)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#80]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#88]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#96]; ldr d9,[x4,#104]; ldr d3,[x4,#112]; ldr d4,[x4,#120]\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#128]\n\t" FMA_3V(16, 17, 23, 0, 0, 1, 8, 9, 8)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(24, 30, 31, 1, 2, 2, 9, 8, 9)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#136]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#144]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#152]; ldr d9,[x4,#160]; ldr d3,[x4,#168]; ldr d4,[x4,#176]\n\t"\
  "add x4,x4,#224; prfm pldl1keep,[x2,#64]; sub w5,w5,#4\n\t"\
  FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#-40]\n\t" FMA_3V(16, 17, 23, 0, 0, 1, 8, 9, 8)\
  "ld1r {v0.2s},[x0],#4; cmp w5,#5\n\t"\
  FMA_3V(24, 30, 31, 1, 2, 2, 9, 8, 9)

#define KERNEL_M3N14_MAIN2 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-16]; ldr d9,[x4,#-8]; ldr d3,[x4]; ldr d4,[x4,#8]\n\t"\
  FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(16, 17, 23, 0, 0, 1, 8, 9, 8)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(24, 30, 31, 1, 2, 2, 9, 8, 9)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#40]; ldr d9,[x4,#48]; ldr d3,[x4,#56]\n\t"\
  "ldr d4,[x4,#64]; add x4,x4,#112\n\t"\
  "sub w5,w5,#2\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#-40]\n\t" FMA_3V(16, 17, 23, 0, 0, 1, 8, 9, 8)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(24, 30, 31, 1, 2, 2, 9, 8, 9)

#define KERNEL_M3N14_TAIL2 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-16]; ldr d9,[x4,#-8]; ldr d3,[x4]; ldr d4,[x4,#8]\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(16, 17, 23, 0, 0, 1, 8, 9, 8)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(24, 30, 31, 1, 2, 2, 9, 8, 9)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#40]; ldr d9,[x4,#48]; add x4,x4,#56\n\t"\
  "sub w5,w5,#2\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(16, 17, 23, 0, 0, 1, 8, 9, 8)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(24, 30, 31, 1, 2, 2, 9, 8, 9)

#define KERNEL_M3N14_TAIL1 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-24]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-16]; ldr d9,[x4,#-8]; sub w5,w5,#1\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(16, 17, 23, 0, 0, 1, 8, 9, 8)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(24, 30, 31, 1, 2, 2, 9, 8, 9)


/* acc layout for m3n16 kernel */
/* m0n0 v8 v9 v10 v11 v12 v13 v14 v15 m0n16 */
/* m1n0 v16 v17 v18 v19 v20 v21 v22 v23 m1n16 */
/* m2n0 v24 v25 v26 v27 v28 v29 v30 v31 m2n16 */
/* b-holder layout for m3n16 kernel */
/* n0 v3 v4 v5 v6 v7 v5 v6 v7 n16 */
/* a-holder layout for m3n16 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2 */

#define INIT_M3N16 \
  INIT_3V(8, 16, 24) INIT_3V(9, 17, 25) INIT_3V(10, 18, 26)\
  INIT_3V(11, 19, 27) INIT_3V(12, 20, 28) INIT_3V(13, 21, 29)\
  INIT_3V(14, 22, 30) INIT_3V(15, 23, 31)

#define SAVE_M3N16(mode) \
  UNIT_SAVE_M3N2_##mode(8, 16, 24) UNIT_SAVE_M3N2_##mode(9, 17, 25)\
  UNIT_SAVE_M3N2_##mode(10, 18, 26) UNIT_SAVE_M3N2_##mode(11, 19, 27)\
  UNIT_SAVE_M3N2_##mode(12, 20, 28) UNIT_SAVE_M3N2_##mode(13, 21, 29)\
  UNIT_SAVE_M3N2_##mode(14, 22, 30) UNIT_SAVE_M3N2_##mode(15, 23, 31)

#define KERNEL_M3N16_PRELOAD1 \
  "ld1r {v0.2s},[x0],#4\n\t"\
  "ldr d3,[x4]; ldr d4,[x4,#8]; ldr d5,[x4,#16]; add x4,x4,#64\n\t"

#define KERNEL_M3N16_MAIN4 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-40]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; ldr d3,[x4]; ldr d4,[x4,#8]\n\t"\
  FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(14, 15, 22, 0, 0, 1, 6, 7, 6)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(23, 30, 31, 1, 2, 2, 7, 6, 7)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#48]; ldr d7,[x4,#56]; ldr d3,[x4,#64]; ldr d4,[x4,#72]\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#80]\n\t" FMA_3V(14, 15, 22, 0, 0, 1, 6, 7, 6)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(23, 30, 31, 1, 2, 2, 7, 6, 7)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#88]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#96]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#104]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#112]; ldr d7,[x4,#120]; ldr d3,[x4,#128]; ldr d4,[x4,#136]\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#144]\n\t" FMA_3V(14, 15, 22, 0, 0, 1, 6, 7, 6)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(23, 30, 31, 1, 2, 2, 7, 6, 7)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#152]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#160]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#168]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#176]; ldr d7,[x4,#184]; ldr d3,[x4,#192]; ldr d4,[x4,#200]\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#208]; add x4,x4,#256\n\t" FMA_3V(14, 15, 22, 0, 0, 1, 6, 7, 6)\
  "sub w5,w5,#4; ld1r {v0.2s},[x0],#4; cmp w5,#5\n\t"\
  FMA_3V(23, 30, 31, 1, 2, 2, 7, 6, 7)

#define KERNEL_M3N16_MAIN2 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-40]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; ldr d3,[x4]; ldr d4,[x4,#8]\n\t"\
  FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(14, 15, 22, 0, 0, 1, 6, 7, 6)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(23, 30, 31, 1, 2, 2, 7, 6, 7)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#48]; ldr d7,[x4,#56]; ldr d3,[x4,#64]; ldr d4,[x4,#72]\n\t"\
  FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#80]; add x4,x4,#128\n\t" FMA_3V(14, 15, 22, 0, 0, 1, 6, 7, 6)\
  "ld1r {v0.2s},[x0],#4; sub w5,w5,#2\n\t" FMA_3V(23, 30, 31, 1, 2, 2, 7, 6, 7)

#define KERNEL_M3N16_TAIL2 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-40]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; ldr d3,[x4]; ldr d4,[x4,#8]\n\t"\
  FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(14, 15, 22, 0, 0, 1, 6, 7, 6)\
  "ld1r {v0.2s},[x0],#4\n\t" FMA_3V(23, 30, 31, 1, 2, 2, 7, 6, 7)\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#48]; ldr d7,[x4,#56]\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "add x4,x4,#64; prfm pldl1keep,[x8]\n\t" FMA_3V(14, 15, 22, 0, 0, 1, 6, 7, 6)\
  "sub w5,w5,#2; prfm pldl1keep,[x9]\n\t" FMA_3V(23, 30, 31, 1, 2, 2, 7, 6, 7)

#define KERNEL_M3N16_TAIL1 \
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-40]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; sub w5,w5,#1\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(14, 15, 22, 0, 0, 1, 6, 7, 6)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(23, 30, 31, 1, 2, 2, 7, 6, 7)

FUNC_PACK0(4, 10)

FUNC_PACK0(4, 12)

FUNC_PACK0(3, 14)

FUNC_PACK0(3, 16)

/* macro for GEMM with packing pattern NO.#4 */
/* mdim = 3, 4; ndim = 9, 11, 13, 15, 17, 18 */
#define FUNC_PACK4(mdim, ndim) \
static inline void sgemm_skinny1_a35_m##mdim##n##ndim(\
  const float * __restrict__ a_ptr, const float * __restrict__ b_scr,\
  float * __restrict__ c_ptr, uint32_t K, uint32_t LDA, uint32_t LDC,\
  uint8_t c_rowmajor, const float * __restrict__ beta_addr) {\
  __asm__ __volatile__(\
    "mov x4,%[b_scr]\n\t"\
    "mov x0,%[a_ptr]; add x1,%[a_ptr],%w[LDA],UXTW #2\n\t"\
    "add x2,%[a_ptr],%w[LDA],UXTW #3; add x3,x1,%w[LDA],UXTW #3\n\t"\
    "add x8,x0,%w[LDA],UXTW #4; add x9,x1,%w[LDA],UXTW #4\n\t"\
    "add x10,x2,%w[LDA],UXTW #4; add x11,x3,%w[LDA],UXTW #4\n\t"\
    "mov w5,%w[K]\n\t"\
    INIT_M##mdim##N##ndim\
    "cmp w5,#2; b.lt 4f\n\t"\
    KERNEL_M##mdim##N##ndim##_PRELOAD2\
    "cmp w5,#6; b.lt 1f\n\t"\
    ".balign 16; 8:\n\t"\
    KERNEL_M##mdim##N##ndim##_MAIN4 "b.ge 8b\n\t"\
    "1:\n\t"\
    "cmp w5,#4; b.lt 2f\n\t"\
    KERNEL_M##mdim##N##ndim##_TAIL4 "b 4f\n\t"\
    "2:\n\t"\
    KERNEL_M##mdim##N##ndim##_TAIL2\
    "4:\n\t"\
    "cmp w5,#1; b.lt 5f\n\t"\
    KERNEL_M##mdim##N##ndim##_FIN1\
    "5:\n\t"\
    "cmp %w[c_rowmajor],#0; b.eq 6f\n\t"\
    INIT_SAVE_M##mdim##_CR SAVE_M##mdim##N##ndim(CR) "b 7f\n\t"\
    "6:\n\t"\
    INIT_SAVE_CC SAVE_M##mdim##N##ndim(CC)\
    "7:\n\t"\
   ::[a_ptr]"r"(a_ptr), [b_scr]"r"(b_scr), [c_ptr]"r"(c_ptr),\
     [LDA]"r"(LDA), [LDC]"r"(LDC), [K]"r"(K),\
     [beta_addr]"r"(beta_addr), [c_rowmajor]"r"(c_rowmajor)\
   :"cc","memory","x0","x1","x2","x3","x4","x5",\
    "x8","x9","x10","x11","x12","x13","x14","x15",\
    "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",\
    "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21",\
    "v22","v23","v24","v25","v26","v27","v28","v29","v30","v31");\
}

/* acc layout for m4n9 kernel */
/* m0n0 v12 v13 v14 v15 v16_h m0n9 */
/* m1n0 v17 v18 v19 v20 v21_h m1n9 */
/* m2n0 v22 v23 v24 v25 v26_h m2n9 */
/* m3n0 v27 v28 v29 v30 v31_h m3n9 */
/* b-holder layout for m4n9 kernel */
/* n0 v4 v5 v6 v7 v8(s) n9 */
/* a-holder layout for m4n9 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2, a_ptr4->v3 */

#define INIT_M4N9 \
  INIT_4V(12, 17, 22, 27) INIT_4V(13, 18, 23, 28)\
  INIT_4V(14, 19, 24, 29) INIT_4V(15, 20, 25, 30)\
  INIT_4V(16, 21, 26, 31)

#define KERNEL_M4N9_PRELOAD2 \
  "ldr d0,[x0],#8\n\t"\
  "ldr d4,[x4]; ldr d5,[x4,#8]; ldr d6,[x4,#16]; add x4,x4,#72\n\t"

#define KERNEL_M4N9_MAIN4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-48]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#-40]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4,#-32]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "rev64 v0.2s,v0.2s; ldr d5,[x4,#-24]; ldr d6,[x4,#-16]; sub w5,w5,#4\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t" FMA_3V(30, 31, 12, 3, 3, 0, 7, 8, 4)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(13, 14, 17, 0, 0, 1, 5, 6, 4)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 22, 1, 1, 2, 5, 6, 4)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(23, 24, 27, 2, 2, 3, 5, 6, 4)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(28, 29, 15, 3, 3, 0, 5, 6, 7)\
  "ldr d0,[x0],#8; ldr d4,[x4]; ldr d5,[x4,#8]; ldr d6,[x4,#16]; cmp w5,#6\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t" FMA_3V(20, 25, 30, 1, 2, 3, 7, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#32]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4,#40]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "rev64 v0.2s,v0.2s; ldr d5,[x4,#48]; ldr d6,[x4,#56]; add x4,x4,#144\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t" FMA_3V(30, 31, 12, 3, 3, 0, 7, 8, 4)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(13, 14, 17, 0, 0, 1, 5, 6, 4)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 22, 1, 1, 2, 5, 6, 4)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(23, 24, 27, 2, 2, 3, 5, 6, 4)\
  "ldr d7,[x4,#-80]\n\t" FMA_3V(28, 29, 15, 3, 3, 0, 5, 6, 7)\
  "ldr d0,[x0],#8; ldr d4,[x4,#-72]; ldr d5,[x4,#-64]; ldr d6,[x4,#-56]\n\t"\
  "prfm pldl1keep,[x3,#64]\n\t" FMA_3V(20, 25, 30, 1, 2, 3, 7, 7, 7)

#define KERNEL_M4N9_TAIL4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-48]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#-40]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4,#-32]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "rev64 v0.2s,v0.2s; ldr d5,[x4,#-24]; ldr d6,[x4,#-16]; sub w5,w5,#4\n\t"\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(30, 31, 12, 3, 3, 0, 7, 8, 4)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(13, 14, 17, 0, 0, 1, 5, 6, 4)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 22, 1, 1, 2, 5, 6, 4)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(23, 24, 27, 2, 2, 3, 5, 6, 4)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(28, 29, 15, 3, 3, 0, 5, 6, 7)\
  "ldr d0,[x0],#8; ldr d4,[x4]; ldr d5,[x4,#8]; ldr d6,[x4,#16]\n\t"\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(20, 25, 30, 1, 2, 3, 7, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#32]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4,#40]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "rev64 v0.2s,v0.2s; ldr d5,[x4,#48]; ldr d6,[x4,#56]; add x4,x4,#72\n\t"\
  "prfm pldl1keep,[x10]\n\t" FMA_3V(30, 31, 12, 3, 3, 0, 7, 8, 4)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(13, 14, 17, 0, 0, 1, 5, 6, 4)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 22, 1, 1, 2, 5, 6, 4)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(23, 24, 27, 2, 2, 3, 5, 6, 4)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(28, 29, 15, 3, 3, 0, 5, 6, 7)\
  "prfm pldl1keep,[x11]\n\t" FMA_3V(20, 25, 30, 1, 2, 3, 7, 7, 7)

#define KERNEL_M4N9_TAIL2 \
  "ldr d1,[x1],#8\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-48]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr d8,[x4,#-40]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "ldr d4,[x4,#-32]\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "rev64 v0.2s,v0.2s; ldr d5,[x4,#-24]; ldr d6,[x4,#-16]; sub w5,w5,#2\n\t"\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(30, 31, 12, 3, 3, 0, 7, 8, 4)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(13, 14, 17, 0, 0, 1, 5, 6, 4)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 22, 1, 1, 2, 5, 6, 4)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(23, 24, 27, 2, 2, 3, 5, 6, 4)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(28, 29, 15, 3, 3, 0, 5, 6, 7)\
  "prfm pldl1keep,[x9]; prfm pldl1keep,[x10]\n\t"\
  "prfm pldl1keep,[x11]\n\t" FMA_3V(20, 25, 30, 1, 2, 3, 7, 7, 7)

#define KERNEL_M4N9_FIN1 \
  "ld1r {v0.2s},[x0],#4; ldr d4,[x4]; ldr d5,[x4,#8]; ldr d6,[x4,#16]\n\t"\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(12, 13, 14, 0, 0, 0, 4, 5, 6)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(17, 18, 19, 1, 1, 1, 4, 5, 6)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(22, 23, 24, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(27, 28, 29, 3, 3, 3, 4, 5, 6)\
  "ldr s8,[x4,#32]\n\t" FMA_3V(15, 20, 25, 0, 1, 2, 7, 7, 7)\
  "add x4,x4,#36\n\t" FMA_3V(16, 21, 26, 0, 1, 2, 8, 8, 8)\
  "fmla v30.2s,v3.2s,v7.2s; fmla v31.2s,v3.2s,v8.2s\n\t"

#define SAVE_M4N9(mode) \
  UNIT_SAVE_M4N2_##mode(12, 17, 22, 27) UNIT_SAVE_M4N2_##mode(13, 18, 23, 28)\
  UNIT_SAVE_M4N2_##mode(14, 19, 24, 29) UNIT_SAVE_M4N2_##mode(15, 20, 25, 30)\
  UNIT_SAVE_M4N1_##mode(16, 21, 26, 31)


/* acc layout for m4n11 kernel */
/* m0n0 v8 v9 v10 v11 v12 v13_h m0n11 */
/* m1n0 v14 v15 v16 v17 v18 v19_h m1n11 */
/* m2n0 v20 v21 v22 v23 v24 v25_h m2n11 */
/* m3n0 v26 v27 v28 v29 v30 v31_h m3n11 */
/* b-holder layout for m4n11 kernel */
/* n0 v4 v5 v6/v7 v7/v6 v5/v7 v6(s) n11 */
/* a-holder layout for m4n11 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2, a_ptr4->v3 */

#define INIT_M4N11 \
  INIT_4V(8, 14, 20, 26) INIT_4V(9, 15, 21, 27)\
  INIT_4V(10, 16, 22, 28) INIT_4V(11, 17, 23, 29)\
  INIT_4V(12, 18, 24, 30) INIT_4V(13, 19, 25, 31)

#define KERNEL_M4N11_PRELOAD2 \
  "ldr d0,[x0],#8\n\t"\
  "ldr d4,[x4]; ldr d5,[x4,#8]; ldr d6,[x4,#16]; add x4,x4,#88\n\t"

#define KERNEL_M4N11_MAIN4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-64]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-56]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-48]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ldr d5,[x4,#-32]; ldr d7,[x4,#-24]; prfm pldl1keep,[x0,#64]\n\t"\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#-16]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d4,[x4]\n\t" FMA_3V(11, 12, 18, 0, 0, 1, 6, 7, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; prfm pldl1keep,[x1,#64]; sub w5,w5,#4\n\t"\
  "ldr d6,[x4,#16]; fmla v24.2s,v2.2s,v7.2s; fmla v30.2s,v3.2s,v7.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#32]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#40]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#48]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ldr d5,[x4,#56]; ldr d7,[x4,#64]; prfm pldl1keep,[x2,#64]\n\t"\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#72]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d7,[x4,#80]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d4,[x4,#88]\n\t" FMA_3V(11, 12, 18, 0, 0, 1, 6, 7, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#96]; prfm pldl1keep,[x3,#64]; cmp w5,#6\n\t"\
  "ldr d6,[x4,#104]; add x4,x4,#176\n\t"\
  "fmla v24.2s,v2.2s,v7.2s; fmla v30.2s,v3.2s,v7.2s\n\t"

#define KERNEL_M4N11_TAIL4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-64]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-56]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-48]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ldr d5,[x4,#-32]; ldr d7,[x4,#-24]; prfm pldl1keep,[x8]\n\t"\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#-16]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "ldr d4,[x4]\n\t" FMA_3V(11, 12, 18, 0, 0, 1, 6, 7, 7)\
  "ldr d0,[x0],#8; ldr d5,[x4,#8]; prfm pldl1keep,[x9]; sub w5,w5,#4\n\t"\
  "ldr d6,[x4,#16]; fmla v24.2s,v2.2s,v7.2s; fmla v30.2s,v3.2s,v7.2s\n\t"\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#24]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#32]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#40]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#48]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ldr d5,[x4,#56]; ldr d7,[x4,#64]; prfm pldl1keep,[x10]\n\t"\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#72]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d7,[x4,#80]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "prfm pldl1keep,[x11]\n\t" FMA_3V(11, 12, 18, 0, 0, 1, 6, 7, 7)\
  "add x4,x4,#88; fmla v24.2s,v2.2s,v7.2s; fmla v30.2s,v3.2s,v7.2s\n\t"

#define KERNEL_M4N11_TAIL2 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ldr d2,[x2],#8\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ldr d3,[x3],#8\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-64]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-56]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr d6,[x4,#-48]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  "ldr d5,[x4,#-32]; ldr d7,[x4,#-24]; prfm pldl1keep,[x8]\n\t"\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 7)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 7)\
  "rev64 v3.2s,v3.2s\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 7)\
  "ldr d6,[x4,#-16]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 7)\
  "ldr d7,[x4,#-8]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 6, 6, 6)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(11, 12, 18, 0, 0, 1, 6, 7, 7)\
  "prfm pldl1keep,[x10]; sub w5,w5,#2\n\t"\
  "fmla v24.2s,v2.2s,v7.2s; fmla v30.2s,v3.2s,v7.2s\n\t"\
  "prfm pldl1keep,[x11]\n\t"

#define KERNEL_M4N11_FIN1 \
  "ld1r {v0.2s},[x0],#4; ldr d4,[x4]; ldr d5,[x4,#8]\n\t"\
  "ldr d6,[x4,#16]; add x4,x4,#44\n\t"\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 4, 5, 6)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(14, 15, 16, 1, 1, 1, 4, 5, 6)\
  "ld1r {v3.2s},[x3],#4\n\t" FMA_3V(20, 21, 22, 2, 2, 2, 4, 5, 6)\
  "ldr d7,[x4,#-20]\n\t" FMA_3V(26, 27, 28, 3, 3, 3, 4, 5, 6)\
  "ldr d5,[x4,#-12]\n\t" FMA_3V(17, 23, 29, 1, 2, 3, 7, 7, 7)\
  "ldr s6,[x4,#-4]\n\t" FMA_3V(18, 24, 30, 1, 2, 3, 5, 5, 5)\
  FMA_3V(11, 12, 13, 0, 0, 0, 7, 5, 6)\
  FMA_3V(19, 25, 31, 1, 2, 3, 6, 6, 6)

#define SAVE_M4N11(mode) \
  UNIT_SAVE_M4N2_##mode(8, 14, 20, 26) UNIT_SAVE_M4N2_##mode(9, 15, 21, 27)\
  UNIT_SAVE_M4N2_##mode(10, 16, 22, 28) UNIT_SAVE_M4N2_##mode(11, 17, 23, 29)\
  UNIT_SAVE_M4N2_##mode(12, 18, 24, 30) UNIT_SAVE_M4N1_##mode(13, 19, 25, 31)


/* acc layout for m3n13 kernel */
/* m0n0 v11 v12 v13 v14 v15 v16 v17_h m0n13 */
/* m1n0 v18 v19 v20 v21 v22 v23 v24_h m1n13 */
/* m2n0 v25 v26 v27 v28 v29 v30 v31_h m2n13 */
/* b-holder layout for m3n13 kernel */
/* n0 v3 v4 v5 v6 v7 v8 v9(s) n13 */
/* a-holder layout for m3n13 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2 */

#define INIT_M3N13 \
  INIT_3V(11, 18, 25) INIT_3V(12, 19, 26) INIT_3V(13, 20, 27)\
  INIT_3V(14, 21, 28) INIT_3V(15, 22, 29) INIT_3V(16, 23, 30)\
  INIT_3V(17, 24, 31)

#define KERNEL_M3N13_PRELOAD2 \
  "ldr d0,[x0],#8\n\t"\
  "ldr d3,[x4]; ldr d4,[x4,#8]; ldr d5,[x4,#16]; add x4,x4,#104\n\t"

#define KERNEL_M3N13_MAIN4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-80]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-72]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-64]; ldr d9,[x4,#-56]; prfm pldl1keep,[x0,#64]\n\t"\
  "ldr d3,[x4,#-48]; sub w5,w5,#4\n\t"\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#-32]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-16]; ldr d8,[x4,#-8]; prfm pldl1keep,[x1,#64]\n\t"\
  "ldr d3,[x4]; cmp w5,#6\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(15, 22, 16, 0, 1, 0, 7, 7, 8)\
  "ldr d0,[x0],#8\n\t" FMA_3V(29, 23, 30, 2, 1, 2, 7, 8, 8)\
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#40]; ldr d9,[x4,#48]; ldr d3,[x4,#56]; ldr d4,[x4,#64]\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#72]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#80]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#88]; ldr d8,[x4,#96]; ldr d3,[x4,#104]; ldr d4,[x4,#112]\n\t"\
  "add x4,x4,#208\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-88]\n\t" FMA_3V(15, 22, 16, 0, 1, 0, 7, 7, 8)\
  "ldr d0,[x0],#8\n\t" FMA_3V(29, 23, 30, 2, 1, 2, 7, 8, 8)

#define KERNEL_M3N13_TAIL4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-80]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-72]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-64]; ldr d9,[x4,#-56]; prfm pldl1keep,[x3]\n\t"\
  "ldr d3,[x4,#-48]; sub w5,w5,#4\n\t"\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#-32]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-16]; ldr d8,[x4,#-8]; prfm pldl1keep,[x8]\n\t"\
  "ldr d3,[x4]\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(15, 22, 16, 0, 1, 0, 7, 7, 8)\
  "ldr d0,[x0],#8\n\t" FMA_3V(29, 23, 30, 2, 1, 2, 7, 8, 8)\
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#40]; ldr d9,[x4,#48]; ldr d3,[x4,#56]; ldr d4,[x4,#64]\n\t"\
  FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#72]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#80]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#88]; ldr d8,[x4,#96]\n\t"\
  "add x4,x4,#104\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(15, 22, 16, 0, 1, 0, 7, 7, 8)\
  FMA_3V(29, 23, 30, 2, 1, 2, 7, 8, 8)

#define KERNEL_M3N13_TAIL2 \
  "ldr d1,[x1],#8\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-80]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-72]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-64]; ldr d9,[x4,#-56]\n\t"\
  "ldr d3,[x4,#-48]; sub w5,w5,#2\n\t"\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr d5,[x4,#-32]\n\t" FMA_3V(16, 23, 17, 0, 1, 0, 8, 8, 9)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 24, 31, 2, 1, 2, 8, 9, 9)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-24]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-16]; ldr d8,[x4,#-8]\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(15, 22, 16, 0, 1, 0, 7, 7, 8)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(29, 23, 30, 2, 1, 2, 7, 8, 8)

#define KERNEL_M3N13_FIN1 \
  "ld1r {v0.2s},[x0],#4; ldr d3,[x4]\n\t"\
  "ldr d4,[x4,#8]; ldr d5,[x4,#16]; add x4,x4,#52\n\t"\
  "ld1r {v1.2s},[x1],#4\n\t" FMA_3V(11, 12, 13, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(18, 19, 20, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-28]\n\t" FMA_3V(25, 26, 27, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-20]\n\t" FMA_3V(14, 21, 28, 0, 1, 2, 6, 6, 6)\
  "ldr d8,[x4,#-12]\n\t" FMA_3V(15, 22, 29, 0, 1, 2, 7, 7, 7)\
  "ldr s9,[x4,#-4]\n\t" FMA_3V(16, 23, 30, 0, 1, 2, 8, 8, 8)\
  FMA_3V(17, 24, 31, 0, 1, 2, 9, 9, 9)

#define SAVE_M3N13(mode) \
  UNIT_SAVE_M3N2_##mode(11, 18, 25) UNIT_SAVE_M3N2_##mode(12, 19, 26)\
  UNIT_SAVE_M3N2_##mode(13, 20, 27) UNIT_SAVE_M3N2_##mode(14, 21, 28)\
  UNIT_SAVE_M3N2_##mode(15, 22, 29) UNIT_SAVE_M3N2_##mode(16, 23, 30)\
  UNIT_SAVE_M3N1_##mode(17, 24, 31)


/* acc layout for m3n15 kernel */
/* m0n0 v8 v9 v10 v11 v12 v13 v14 v15_h m0n15 */
/* m1n0 v16 v17 v18 v19 v20 v21 v22 v23_h m1n15 */
/* m2n0 v24 v25 v26 v27 v28 v29 v30 v31_h m2n15 */
/* b-holder layout for m3n15 kernel */
/* n0 v3 v4 v5 v6 v7/v5 v5/v6 v6/v7 v7(s) n15 */
/* a-holder layout for m3n15 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2 */

#define INIT_M3N15 \
  INIT_3V(8, 16, 24) INIT_3V(9, 17, 25) INIT_3V(10, 18, 26)\
  INIT_3V(11, 19, 27) INIT_3V(12, 20, 28) INIT_3V(13, 21, 29)\
  INIT_3V(14, 22, 30) INIT_3V(15, 23, 31)

#define KERNEL_M3N15_PRELOAD2 \
  "ldr d0,[x0],#8\n\t"\
  "ldr d3,[x4]; ldr d4,[x4,#8]; ldr d5,[x4,#16]; add x4,x4,#120\n\t"

#define KERNEL_M3N15_MAIN4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-96]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-88]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-80]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-72]; ldr d7,[x4,#-64]; ldr d3,[x4,#-56]; ldr d4,[x4,#-48]\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#-40]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; ldr d3,[x4]; ldr d4,[x4,#8]\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(13, 21, 14, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8\n\t" FMA_3V(29, 22, 30, 2, 1, 2, 6, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#48]; ldr d7,[x4,#56]; ldr d3,[x4,#64]; ldr d4,[x4,#72]\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#80]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#88]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d5,[x4,#96]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d6,[x4,#104]; ldr d7,[x4,#112]; ldr d3,[x4,#120]\n\t"\
  "ldr d4,[x4,#128]; sub w5,w5,#4\n\t"\
  FMA_3V(12, 20, 28, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#136]; add x4,x4,#240\n\t"\
  FMA_3V(13, 21, 14, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8; cmp w5,#6\n\t"\
  FMA_3V(29, 22, 30, 2, 1, 2, 6, 7, 7)

#define KERNEL_M3N15_TAIL4 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-96]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-88]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-80]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-72]; ldr d7,[x4,#-64]; ldr d3,[x4,#-56]; ldr d4,[x4,#-48]\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#-40]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]; ldr d3,[x4]; ldr d4,[x4,#8]\n\t"\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#16]\n\t" FMA_3V(13, 21, 14, 0, 1, 0, 6, 6, 7)\
  "ldr d0,[x0],#8\n\t" FMA_3V(29, 22, 30, 2, 1, 2, 6, 7, 7)\
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#24]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#32]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#40]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#48]; ldr d7,[x4,#56]; ldr d3,[x4,#64]; ldr d4,[x4,#72]\n\t"\
  FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#80]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#88]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d5,[x4,#96]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d6,[x4,#104]; ldr d7,[x4,#112]\n\t"\
  "sub w5,w5,#4\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 5, 5, 5)\
  "prfm pldl1keep,[x9]; add x4,x4,#120\n\t"\
  FMA_3V(13, 21, 14, 0, 1, 0, 6, 6, 7)\
  FMA_3V(29, 22, 30, 2, 1, 2, 6, 7, 7)

#define KERNEL_M3N15_TAIL2 \
  "ldr d1,[x1],#8\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ldr d2,[x2],#8\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-96]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-88]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-80]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-72]; ldr d7,[x4,#-64]; ldr d3,[x4,#-56]; ldr d4,[x4,#-48]\n\t"\
  "prfm pldl1keep,[x3]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr d5,[x4,#-40]\n\t" FMA_3V(14, 22, 15, 0, 1, 0, 6, 6, 7)\
  "rev64 v0.2s,v0.2s\n\t" FMA_3V(30, 23, 31, 2, 1, 2, 6, 7, 7)\
  "rev64 v1.2s,v1.2s\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "rev64 v2.2s,v2.2s\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-32]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d5,[x4,#-24]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d6,[x4,#-16]; ldr d7,[x4,#-8]\n\t"\
  "prfm pldl1keep,[x8]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 5, 5, 5)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(13, 21, 14, 0, 1, 0, 6, 6, 7)\
  "sub w5,w5,#2\n\t" FMA_3V(29, 22, 30, 2, 1, 2, 6, 7, 7)

#define KERNEL_M3N15_FIN1 \
  "ld1r {v0.2s},[x0],#4; ldr d3,[x4]; ldr d4,[x4,#8]; ldr d5,[x4,#16]\n\t"\
  "ld1r {v1.2s},[x1],#4; add x4,x4,#60\n\t" FMA_3V(8, 9, 10, 0, 0, 0, 3, 4, 5)\
  "ld1r {v2.2s},[x2],#4\n\t" FMA_3V(16, 17, 18, 1, 1, 1, 3, 4, 5)\
  "ldr d6,[x4,#-36]\n\t" FMA_3V(24, 25, 26, 2, 2, 2, 3, 4, 5)\
  "ldr d7,[x4,#-28]\n\t" FMA_3V(11, 19, 27, 0, 1, 2, 6, 6, 6)\
  "ldr d5,[x4,#-20]\n\t" FMA_3V(12, 20, 28, 0, 1, 2, 7, 7, 7)\
  "ldr d6,[x4,#-12]\n\t" FMA_3V(13, 21, 29, 0, 1, 2, 5, 5, 5)\
  "ldr s7,[x4,#-4]\n\t" FMA_3V(14, 22, 30, 0, 1, 2, 6, 6, 6)\
  FMA_3V(15, 23, 31, 0, 1, 2, 7, 7, 7)

#define SAVE_M3N15(mode) \
  UNIT_SAVE_M3N2_##mode(8, 16, 24) UNIT_SAVE_M3N2_##mode(9, 17, 25)\
  UNIT_SAVE_M3N2_##mode(10, 18, 26) UNIT_SAVE_M3N2_##mode(11, 19, 27)\
  UNIT_SAVE_M3N2_##mode(12, 20, 28) UNIT_SAVE_M3N2_##mode(13, 21, 29)\
  UNIT_SAVE_M3N2_##mode(14, 22, 30) UNIT_SAVE_M3N1_##mode(15, 23, 31)


/* acc layout for m3n17 kernel */
/* m0n0 v5 v6 v7 v8 v9 v10 v11 v12 v13_h m0n17 */
/* m1n0 v14 v15 v16 v17 v18 v19 v20 v21 v22_h m1n17 */
/* m2n0 v23 v24 v25 v26 v27 v28 v29 v30 v31_h m2n17 */
/* b-holder layout for m3n17 kernel */
/* n0 v3-4 alt n17 */
/* a-holder layout for m3n17 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2 */

#define INIT_M3N17 \
  INIT_3V(5, 14, 23) INIT_3V(6, 15, 24) INIT_3V(7, 16, 25)\
  INIT_3V(8, 17, 26) INIT_3V(9, 18, 27) INIT_3V(10, 19, 28)\
  INIT_3V(11, 20, 29) INIT_3V(12, 21, 30) INIT_3V(13, 22, 31)

#define KERNEL_M3N17_PRELOAD2 \
  "ldr d3,[x4],#136\n\t"

#define KERNEL_M3N17_MAIN4 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "prfm pldl1keep,[x0,#64]\n\t"\
  "ldr d4,[x4,#-128]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-120]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-112]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-104]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-96]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-88]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-80]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-72]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-64]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "prfm pldl1keep,[x1,#64]\n\t"\
  "ldr d3,[x4,#-56]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-48]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-40]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-32]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-24]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-16]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-8]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)\
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "ldr d3,[x4,#8]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#16]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#24]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#32]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#40]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#48]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#56]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#64]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#72]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 4, 4, 4)\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t"\
  "ldr d4,[x4,#80]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#88]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#96]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#104]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#112]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#120]; sub w5,w5,#4\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#128]; cmp w5,#6\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#136]; add x4,x4,#272\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)

#define KERNEL_M3N17_TAIL4 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "ldr d4,[x4,#-128]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-120]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-112]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-104]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-96]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-88]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-80]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-72]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-64]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "prfm pldl1keep,[x3]\n\t"\
  "ldr d3,[x4,#-56]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-48]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-40]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-32]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-24]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-16]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-8]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)\
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "ldr d3,[x4,#8]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#16]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#24]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#32]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#40]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#48]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#56]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#64]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#72]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 4, 4, 4)\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "prfm pldl1keep,[x8]\n\t"\
  "ldr d4,[x4,#80]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#88]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#96]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#104]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#112]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#120]; sub w5,w5,#4\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#128]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "prfm pldl1keep,[x9]; add x4,x4,#136\n\t"\
  FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)

#define KERNEL_M3N17_TAIL2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "prfm pldl1keep,[x3]\n\t"\
  "ldr d4,[x4,#-128]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-120]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-112]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-104]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-96]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-88]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-80]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-72]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-64]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "sub w5,w5,#2; prfm pldl1keep,[x8]\n\t"\
  "ldr d3,[x4,#-56]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-48]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-40]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-32]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-24]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-16]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-8]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)

#define KERNEL_M3N17_FIN1 \
  "ldr d3,[x4],#68\n\t"\
  "ld1r {v0.2s},[x0],#4; ld1r {v1.2s},[x1],#4; ld1r {v2.2s},[x2],#4\n\t"\
  "ldr d4,[x4,#-60]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-52]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-44]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-36]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-28]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-20]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-12]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr s3,[x4,#-4]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)

#define SAVE_M3N17(mode) \
  UNIT_SAVE_M3N2_##mode(5, 14, 23) UNIT_SAVE_M3N2_##mode(6, 15, 24)\
  UNIT_SAVE_M3N2_##mode(7, 16, 25) UNIT_SAVE_M3N2_##mode(8, 17, 26)\
  UNIT_SAVE_M3N2_##mode(9, 18, 27) UNIT_SAVE_M3N2_##mode(10, 19, 28)\
  UNIT_SAVE_M3N2_##mode(11, 20, 29) UNIT_SAVE_M3N2_##mode(12, 21, 30)\
  UNIT_SAVE_M3N1_##mode(13, 22, 31)


/* acc layout for m3n18 kernel */
/* m0n0 v5 v6 v7 v8 v9 v10 v11 v12 v13 m0n18 */
/* m1n0 v14 v15 v16 v17 v18 v19 v20 v21 v22 m1n18 */
/* m2n0 v23 v24 v25 v26 v27 v28 v29 v30 v31 m2n18 */
/* b-holder layout for m3n18 kernel */
/* n0 v3-4 alt n18 */
/* a-holder layout for m3n18 kernel */
/* a_ptr1->v0, a_ptr2->v1, a_ptr3->v2 */

#define INIT_M3N18 \
  INIT_3V(5, 14, 23) INIT_3V(6, 15, 24) INIT_3V(7, 16, 25)\
  INIT_3V(8, 17, 26) INIT_3V(9, 18, 27) INIT_3V(10, 19, 28)\
  INIT_3V(11, 20, 29) INIT_3V(12, 21, 30) INIT_3V(13, 22, 31)

#define KERNEL_M3N18_PRELOAD2 \
  "ldr d3,[x4],#144\n\t"

#define KERNEL_M3N18_MAIN4 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "prfm pldl1keep,[x0,#64]; sub w5,w5,#4\n\t"\
  "ldr d4,[x4,#-136]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-128]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-120]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-112]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-104]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-96]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-88]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-80]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-72]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "prfm pldl1keep,[x1,#64]; cmp w5,#6\n\t"\
  "ldr d3,[x4,#-64]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-56]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-48]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-32]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-24]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-16]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-8]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 4, 4, 4)\
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "prfm pldl1keep,[x2,#64]\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#16]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#24]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#32]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#40]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#48]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#56]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#64]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#72]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)\
  "add x4,x4,#288\n\t"\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "ldr d3,[x4,#-208]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-200]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-192]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-184]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-176]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-168]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-160]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-152]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-144]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 4, 4, 4)

#define KERNEL_M3N18_TAIL4 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "prfm pldl1keep,[x3]; sub w5,w5,#4\n\t"\
  "ldr d4,[x4,#-136]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-128]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-120]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-112]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-104]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-96]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-88]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-80]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-72]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "prfm pldl1keep,[x8]\n\t"\
  "ldr d3,[x4,#-64]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-56]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-48]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-32]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-24]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-16]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-8]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 4, 4, 4)\
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "ldr d4,[x4,#8]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#16]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#24]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#32]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#40]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#48]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#56]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#64]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#72]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)\
  "add x4,x4,#144\n\t"\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "ldr d3,[x4,#-64]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-56]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-48]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-32]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-24]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-16]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-8]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 4, 4, 4)

#define KERNEL_M3N18_TAIL2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr d2,[x2],#8\n\t"\
  "prfm pldl1keep,[x3]; sub w5,w5,#2\n\t"\
  "ldr d4,[x4,#-136]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-128]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-120]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-112]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-104]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-96]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-88]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-80]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-72]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)\
  "rev64 v0.2s,v0.2s; rev64 v1.2s,v1.2s; rev64 v2.2s,v2.2s\n\t"\
  "prfm pldl1keep,[x8]\n\t"\
  "ldr d3,[x4,#-64]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-56]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-48]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-40]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-32]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-24]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-16]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-8]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 3, 3, 3)\
  "prfm pldl1keep,[x9]\n\t" FMA_3V(13, 22, 31, 0, 1, 2, 4, 4, 4)

#define KERNEL_M3N18_FIN1 \
  "ldr d3,[x4],#72\n\t"\
  "ld1r {v0.2s},[x0],#4; ld1r {v1.2s},[x1],#4; ld1r {v2.2s},[x2],#4\n\t"\
  "ldr d4,[x4,#-64]\n\t" FMA_3V(5, 14, 23, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-56]\n\t" FMA_3V(6, 15, 24, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-48]\n\t" FMA_3V(7, 16, 25, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-40]\n\t" FMA_3V(8, 17, 26, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-32]\n\t" FMA_3V(9, 18, 27, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-24]\n\t" FMA_3V(10, 19, 28, 0, 1, 2, 4, 4, 4)\
  "ldr d4,[x4,#-16]\n\t" FMA_3V(11, 20, 29, 0, 1, 2, 3, 3, 3)\
  "ldr d3,[x4,#-8]\n\t" FMA_3V(12, 21, 30, 0, 1, 2, 4, 4, 4)\
  FMA_3V(13, 22, 31, 0, 1, 2, 3, 3, 3)

#define SAVE_M3N18(mode) \
  UNIT_SAVE_M3N2_##mode(5, 14, 23) UNIT_SAVE_M3N2_##mode(6, 15, 24)\
  UNIT_SAVE_M3N2_##mode(7, 16, 25) UNIT_SAVE_M3N2_##mode(8, 17, 26)\
  UNIT_SAVE_M3N2_##mode(9, 18, 27) UNIT_SAVE_M3N2_##mode(10, 19, 28)\
  UNIT_SAVE_M3N2_##mode(11, 20, 29) UNIT_SAVE_M3N2_##mode(12, 21, 30)\
  UNIT_SAVE_M3N2_##mode(13, 22, 31)


FUNC_PACK4(4, 9)

FUNC_PACK4(4, 11)

FUNC_PACK4(3, 13)

FUNC_PACK4(3, 15)

FUNC_PACK4(3, 17)

FUNC_PACK4(3, 18)


#define INIT_M1N4 \
  float32x2_t cd1, cd2, cd3, cd4;\
  cd1 = cd2 = cd3 = cd4 = vdup_n_f32(0.0f);

#define INIT_M1N5 INIT_M1N4 float32x2_t cd5 = vdup_n_f32(0.0f);

#define INIT_M1N6 INIT_M1N5 float32x2_t cd6 = vdup_n_f32(0.0f);

#define INIT_M1N7 INIT_M1N6 float32x2_t cd7 = vdup_n_f32(0.0f);

#define INIT_M1N8 INIT_M1N7 float32x2_t cd8 = vdup_n_f32(0.0f);

#define INIT_M1N10 INIT_M1N5

#define INIT_M1N12 INIT_M1N6

#define INIT_M1N14 INIT_M1N7

#define INIT_M1N16 INIT_M1N8

#define INIT_M1N9 \
  float32x2_t cd1, cd2, cd3, cd4;\
  cd1 = cd2 = cd3 = cd4 = vdup_n_f32(0.0f);\
  float32x2_t cd0 = vdup_n_f32(0.0f);

#define INIT_M1N11 INIT_M1N10 float32x2_t cd0 = vdup_n_f32(0.0f);

#define INIT_M1N13 INIT_M1N12 float32x2_t cd0 = vdup_n_f32(0.0f);

#define INIT_M1N15 INIT_M1N14 float32x2_t cd0 = vdup_n_f32(0.0f);

#define INIT_M1N17 INIT_M1N16 float32x2_t cd0 = vdup_n_f32(0.0f);

#define INIT_M1N18 INIT_M1N16 float32x2_t cd9 = vdup_n_f32(0.0f);

#define LOAD_4D_B \
  float32x2_t bd1 = vld1_f32(b_ptr);\
  float32x2_t bd2 = vld1_f32(b_ptr + 2);\
  float32x2_t bd3 = vld1_f32(b_ptr + 4);\
  float32x2_t bd4 = vld1_f32(b_ptr + 6);

#define LOAD_5D_B LOAD_4D_B float32x2_t bd5 = vld1_f32(b_ptr + 8);

#define LOAD_6D_B LOAD_5D_B float32x2_t bd6 = vld1_f32(b_ptr + 10);

#define LOAD_7D_B LOAD_6D_B float32x2_t bd7 = vld1_f32(b_ptr + 12);

#define LOAD_8D_B LOAD_7D_B float32x2_t bd8 = vld1_f32(b_ptr + 14);

#define LOAD_9D_B LOAD_8D_B float32x2_t bd9 = vld1_f32(b_ptr + 16);

#define ACC_4D \
  cd1 = vfma_f32(cd1, ad1, bd1);\
  cd2 = vfma_f32(cd2, ad1, bd2);\
  cd3 = vfma_f32(cd3, ad1, bd3);\
  cd4 = vfma_f32(cd4, ad1, bd4);

#define ACC_5D ACC_4D cd5 = vfma_f32(cd5, ad1, bd5);

#define ACC_6D ACC_5D cd6 = vfma_f32(cd6, ad1, bd6);

#define ACC_7D ACC_6D cd7 = vfma_f32(cd7, ad1, bd7);

#define ACC_8D ACC_7D cd8 = vfma_f32(cd8, ad1, bd8);

#define ACC_9D ACC_8D cd9 = vfma_f32(cd9, ad1, bd9);

#define REDUC_4D \
  float cs1 = vpadds_f32(cd1); float cs2 = vpadds_f32(cd2);\
  float cs3 = vpadds_f32(cd3); float cs4 = vpadds_f32(cd4);\

#define REDUC_5D REDUC_4D float cs5 = vpadds_f32(cd5);

#define REDUC_6D REDUC_5D float cs6 = vpadds_f32(cd6);

#define REDUC_7D REDUC_6D float cs7 = vpadds_f32(cd7);

#define REDUC_8D REDUC_7D float cs8 = vpadds_f32(cd8);

#define ACC_4S \
  cs1 += as1 * b_ptr[0]; cs2 += as1 * b_ptr[1];\
  cs3 += as1 * b_ptr[2]; cs4 += as1 * b_ptr[3];\

#define ACC_5S ACC_4S cs5 += as1 * b_ptr[4];

#define ACC_6S ACC_5S cs6 += as1 * b_ptr[5];

#define ACC_7S ACC_6S cs7 += as1 * b_ptr[6];

#define ACC_8S ACC_7S cs8 += as1 * b_ptr[7];

#define UNIT_SAVE_M1N1_CC(cs1) \
  c_ptr[0] = c_ptr[0] * beta + cs1; c_ptr += LDC;

#define UNIT_SAVE_M1N1_CR(cs1) \
  c_ptr[0] = c_ptr[0] * beta + cs1; c_ptr++;

#define UNIT_SAVE_M1N2_CC(cd1) \
  c_ptr[0] = c_ptr[0] * beta + vget_lane_f32(cd1, 0);\
  c_ptr[LDC] = c_ptr[LDC] * beta + vget_lane_f32(cd1, 1);\
  c_ptr += LDC * 2;

#define UNIT_SAVE_M1N2_CR(cd1) \
  cd1 = vfma_n_f32(cd1, vld1_f32(c_ptr), beta);\
  vst1_f32(c_ptr, cd1); c_ptr += 2;

#define SAVE_M1N4(mode) \
  UNIT_SAVE_M1N1_##mode(cs1) UNIT_SAVE_M1N1_##mode(cs2)\
  UNIT_SAVE_M1N1_##mode(cs3) UNIT_SAVE_M1N1_##mode(cs4)\

#define SAVE_M1N5(mode) SAVE_M1N4(mode) UNIT_SAVE_M1N1_##mode(cs5)

#define SAVE_M1N6(mode) SAVE_M1N5(mode) UNIT_SAVE_M1N1_##mode(cs6)

#define SAVE_M1N7(mode) SAVE_M1N6(mode) UNIT_SAVE_M1N1_##mode(cs7)

#define SAVE_M1N8(mode) SAVE_M1N7(mode) UNIT_SAVE_M1N1_##mode(cs8)

#define SAVE_M1N10(mode) \
  UNIT_SAVE_M1N2_##mode(cd1) UNIT_SAVE_M1N2_##mode(cd2)\
  UNIT_SAVE_M1N2_##mode(cd3) UNIT_SAVE_M1N2_##mode(cd4)\
  UNIT_SAVE_M1N2_##mode(cd5)

#define SAVE_M1N12(mode) SAVE_M1N10(mode) UNIT_SAVE_M1N2_##mode(cd6)

#define SAVE_M1N14(mode) SAVE_M1N12(mode) UNIT_SAVE_M1N2_##mode(cd7)

#define SAVE_M1N16(mode) SAVE_M1N14(mode) UNIT_SAVE_M1N2_##mode(cd8)

#define SAVE_M1N18(mode) SAVE_M1N16(mode) UNIT_SAVE_M1N2_##mode(cd9)

#define SAVE_M1N9(mode) \
  UNIT_SAVE_M1N2_##mode(cd1) UNIT_SAVE_M1N2_##mode(cd2)\
  UNIT_SAVE_M1N2_##mode(cd3) UNIT_SAVE_M1N2_##mode(cd4)\
  UNIT_SAVE_M1N1_##mode(cs0)

#define SAVE_M1N11(mode) SAVE_M1N10(mode) UNIT_SAVE_M1N1_##mode(cs0)

#define SAVE_M1N13(mode) SAVE_M1N12(mode) UNIT_SAVE_M1N1_##mode(cs0)

#define SAVE_M1N15(mode) SAVE_M1N14(mode) UNIT_SAVE_M1N1_##mode(cs0)

#define SAVE_M1N17(mode) SAVE_M1N16(mode) UNIT_SAVE_M1N1_##mode(cs0)

#define COMPUTE_M1_PACK3(ndim) \
  for (; k_left > 1; k_left -= 2) {\
    float32x2_t ad1 = vld1_f32(a_ptr); a_ptr += 2;\
    LOAD_##ndim##D_B\
    ACC_##ndim##D\
    b_ptr += 2 * ndim;\
  }\
  REDUC_##ndim##D\
  if (k_left > 0) {\
    float as1 = *a_ptr;\
    ACC_##ndim##S\
  }

#define COMPUTE_M1_PACK0_BASE(ndiv2) \
  for (; k_left > 0; k_left--) {\
    float32x2_t ad1 = vld1_dup_f32(a_ptr); a_ptr++;\
    LOAD_##ndiv2##D_B\
    ACC_##ndiv2##D\
    b_ptr += ndiv2 * 2;\
  }

#define COMPUTE_M1_PACK0_N10 COMPUTE_M1_PACK0_BASE(5)
#define COMPUTE_M1_PACK0_N12 COMPUTE_M1_PACK0_BASE(6)
#define COMPUTE_M1_PACK0_N14 COMPUTE_M1_PACK0_BASE(7)
#define COMPUTE_M1_PACK0_N16 COMPUTE_M1_PACK0_BASE(8)
#define COMPUTE_M1_PACK0(ndim) COMPUTE_M1_PACK0_N##ndim

#define COMPUTE_M1_PACK4_EVEN(ndiv2) \
  for (; k_left > 1; k_left -= 2) {\
    float32x2_t ad1 = vld1_f32(a_ptr); a_ptr += 2;\
    {\
      LOAD_##ndiv2##D_B b_ptr += ndiv2 * 2;\
      ACC_##ndiv2##D\
    }\
    ad1 = vrev64_f32(ad1);\
    LOAD_##ndiv2##D_B b_ptr += ndiv2 * 2;\
    ACC_##ndiv2##D\
  }\
  if (k_left > 0) {\
    float32x2_t ad1 = vld1_dup_f32(a_ptr);\
    LOAD_##ndiv2##D_B\
    ACC_##ndiv2##D\
  }

#define COMPUTE_M1_PACK4_N18 COMPUTE_M1_PACK4_EVEN(9)

#define COMPUTE_M1_PACK4_ODD(ndiv2) \
  for (; k_left > 1; k_left -= 2) {\
    float32x2_t ad1 = vld1_f32(a_ptr); a_ptr += 2;\
    {\
      LOAD_##ndiv2##D_B\
      float32x2_t bd0 = vld1_f32(b_ptr + ndiv2 * 2);\
      b_ptr += ndiv2 * 2 + 2;\
      ACC_##ndiv2##D\
      cd0 = vfma_f32(cd0, ad1, bd0);\
    }\
    ad1 = vrev64_f32(ad1);\
    LOAD_##ndiv2##D_B b_ptr += ndiv2 * 2;\
    ACC_##ndiv2##D\
  }\
  float cs0 = vpadds_f32(cd0);\
  if (k_left > 0) {\
    float32x2_t ad1 = vld1_dup_f32(a_ptr);\
    LOAD_##ndiv2##D_B\
    float bs0 = b_ptr[ndiv2 * 2];\
    ACC_##ndiv2##D\
    cs0 += bs0 * vget_lane_f32(ad1, 0);\
  }

#define COMPUTE_M1_PACK4_N9 COMPUTE_M1_PACK4_ODD(4)
#define COMPUTE_M1_PACK4_N11 COMPUTE_M1_PACK4_ODD(5)
#define COMPUTE_M1_PACK4_N13 COMPUTE_M1_PACK4_ODD(6)
#define COMPUTE_M1_PACK4_N15 COMPUTE_M1_PACK4_ODD(7)
#define COMPUTE_M1_PACK4_N17 COMPUTE_M1_PACK4_ODD(8)

#define COMPUTE_M1_PACK4(ndim) COMPUTE_M1_PACK4_N##ndim    

#define FUNC_EDGE(ndim, pack) \
static inline void sgemm_skinny1_a35_m1n##ndim(\
  const float * __restrict__ a_ptr, const float * __restrict__ b_ptr,\
  float * __restrict__ c_ptr, uint32_t k_left, uint32_t LDC,\
  uint8_t c_rowmajor, float beta) {\
  INIT_M1N##ndim\
  COMPUTE_M1_PACK##pack(ndim)\
  if (c_rowmajor == 0) {\
    SAVE_M1N##ndim(CC)\
  } else {\
    SAVE_M1N##ndim(CR)\
  }\
}

FUNC_EDGE(4, 3)

FUNC_EDGE(5, 3)

FUNC_EDGE(6, 3)

FUNC_EDGE(7, 3)

FUNC_EDGE(8, 3)

FUNC_EDGE(10, 0)

FUNC_EDGE(12, 0)

FUNC_EDGE(14, 0)

FUNC_EDGE(16, 0)

FUNC_EDGE(9, 4)

FUNC_EDGE(11, 4)

FUNC_EDGE(13, 4)

FUNC_EDGE(15, 4)

FUNC_EDGE(17, 4)

FUNC_EDGE(18, 4)

#endif
