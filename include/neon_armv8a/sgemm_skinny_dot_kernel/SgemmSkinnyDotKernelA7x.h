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

#ifndef INCLUDE_A7X_KERNEL
#define INCLUDE_A7X_KERNEL

/* x0 - x3 for a_ptrs */
/* x4 for b_ptr, x5 for k_left */
/* x6 - x9 for a_pref */
/* x12 - x15 for c_tmp */

#define INIT_1V(c1) "movi v"#c1".16b,#0\n\t"

#define INIT_2V(c1, c2) INIT_1V(c1) INIT_1V(c2)

#define INIT_4V(c1, c2, c3, c4) INIT_2V(c1, c2) INIT_2V(c3, c4)

#define INIT_SAVE \
  "ldr s0,[%[beta_addr]]; mov x12,%[c_ptr]\n\t"\
  "add x13,%[c_ptr],%w[LDC],UXTW #2; add x14,%[c_ptr],%w[LDC],UXTW #3\n\t"\
  "add x15,x13,%w[LDC],UXTW #3\n\t"

#define UNIT_SAVE_M4N4_CC(c1, c2, c3, c4) \
  "trn1 v1.4s,v"#c1".4s,v"#c2".4s; trn1 v2.4s,v"#c3".4s,v"#c4".4s\n\t"\
  "trn2 v3.4s,v"#c1".4s,v"#c2".4s; trn2 v4.4s,v"#c3".4s,v"#c4".4s\n\t"\
  "trn1 v"#c1".2d,v1.2d,v2.2d; trn1 v"#c2".2d,v3.2d,v4.2d\n\t"\
  "trn2 v"#c3".2d,v1.2d,v2.2d; trn2 v"#c4".2d,v3.2d,v4.2d\n\t"\
  "ldr q1,[x12]; ldr q2,[x13]; ldr q3,[x14]; ldr q4,[x15]\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]; fmla v"#c2".4s,v2.4s,v0.s[0]\n\t"\
  "fmla v"#c3".4s,v3.4s,v0.s[0]; fmla v"#c4".4s,v4.4s,v0.s[0]\n\t"\
  "str q"#c1",[x12]; prfm pstl2keep,[x12,#32]; add x12,x12,%w[LDC],UXTW #4\n\t"\
  "str q"#c2",[x13]; prfm pstl2keep,[x13,#32]; add x13,x13,%w[LDC],UXTW #4\n\t"\
  "str q"#c3",[x14]; prfm pstl2keep,[x14,#32]; add x14,x14,%w[LDC],UXTW #4\n\t"\
  "str q"#c4",[x15]; prfm pstl2keep,[x15,#32]; add x15,x15,%w[LDC],UXTW #4\n\t"

#define EDGE_SAVE_M4N1_CC(c1, c2, c3, c4) \
  "ldr q1,[x12]\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c2".4s\n\t"\
  "faddp v"#c3".4s,v"#c3".4s,v"#c4".4s\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c3".4s\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]\n\t"\
  "str q"#c1",[x12]; prfm pstl2keep,[x12,#32]\n\t"\
  "add x12,x12,%w[LDC],UXTW #2\n\t"

#define UNIT_SAVE_M4N4_CR(c1, c2, c3, c4) \
  "ldr q1,[x12]; ldr q2,[x13]; ldr q3,[x14]; ldr q4,[x15]\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]; fmla v"#c2".4s,v2.4s,v0.s[0]\n\t"\
  "fmla v"#c3".4s,v3.4s,v0.s[0]; fmla v"#c4".4s,v4.4s,v0.s[0]\n\t"\
  "str q"#c1",[x12],#16; str q"#c2",[x13],#16\n\t"\
  "str q"#c3",[x14],#16; str q"#c4",[x15],#16\n\t"

#define EDGE_SAVE_M4N1_CR(c1, c2, c3, c4) \
  "ldr s1,[x12]; ldr s2,[x13]; ldr s3,[x14]; ldr s4,[x15]\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c2".4s\n\t"\
  "ins v1.s[1],v2.s[0]; ins v3.s[1],v4.s[0]\n\t"\
  "faddp v"#c3".4s,v"#c3".4s,v"#c4".4s\n\t"\
  "ins v1.d[1],v3.d[0]\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c3".4s\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]\n\t"\
  "st1 {v"#c1".s}[0],[x12],#4; st1 {v"#c1".s}[1],[x13],#4\n\t"\
  "st1 {v"#c1".s}[2],[x14],#4; st1 {v"#c1".s}[3],[x15],#4\n\t"

#define FUNC_M4(ndim) \
static inline void sgemm_skinny1_a7x_m4n##ndim(\
  const float * __restrict__ a_ptr, const float * __restrict__ b_scr,\
  float * __restrict__ c_ptr, uint32_t K, uint32_t LDA, uint32_t LDC,\
  uint8_t c_rowmajor, const float * __restrict__ beta_addr) {\
  __asm__ __volatile__ (\
    "mov x0,%[a_ptr]; add x1,%[a_ptr],%w[LDA],UXTW #2\n\t"\
    "add x2,%[a_ptr],%w[LDA],UXTW #3; add x3,x1,%w[LDA],UXTW #3\n\t"\
    "add x6,x0,%w[LDA],UXTW #4; add x7,x1,%w[LDA],UXTW #4\n\t"\
    "add x8,x2,%w[LDA],UXTW #4; add x9,x3,%w[LDA],UXTW #4\n\t"\
    "mov x4,%[b_scr]; mov w5,%w[K]\n\t"\
    INIT_M4N##ndim\
    "cmp w5,#4; b.lt 4f\n\t"\
    KERNEL_M4N##ndim##_PRELOAD4\
    "cmp w5,#20; b.lt 1f\n\t"\
    ".balign 16; 9:\n\t"\
    "prfm pldl2keep,[x6]; add x6,x6,#64\n\t"\
    KERNEL_M4N##ndim##_MAIN4(0, 1, 2, 3, 4, 5, 6, 7)\
    "prfm pldl2keep,[x7]; add x7,x7,#64\n\t"\
    KERNEL_M4N##ndim##_MAIN4(4, 5, 6, 7, 0, 1, 2, 3)\
    "prfm pldl2keep,[x8]; add x8,x8,#64\n\t"\
    KERNEL_M4N##ndim##_MAIN4(0, 1, 2, 3, 4, 5, 6, 7)\
    "prfm pldl2keep,[x9]; add x9,x9,#64\n\t"\
    KERNEL_M4N##ndim##_MAIN4(4, 5, 6, 7, 0, 1, 2, 3)\
    "cmp w5,#20; b.ge 9b; 1:\n\t"\
    "cmp w5,#12; b.lt 2f\n\t"\
    KERNEL_M4N##ndim##_MAIN4(0, 1, 2, 3, 4, 5, 6, 7)\
    KERNEL_M4N##ndim##_MAIN4(4, 5, 6, 7, 0, 1, 2, 3)\
    "2:\n\t"\
    "cmp w5,#8; b.lt 3f\n\t"\
    KERNEL_M4N##ndim##_MAIN4(0, 1, 2, 3, 4, 5, 6, 7)\
    KERNEL_M4N##ndim##_TAIL4(4, 5, 6, 7)\
    "b 4f; 3:\n\t"\
    KERNEL_M4N##ndim##_TAIL4(0, 1, 2, 3)\
    "4:\n\t"\
    "cmp w5,#1; b.lt 6f\n\t"\
    "5:\n\t"\
    KERNEL_M4N##ndim##_TL1 "b.gt 5b\n\t"\
    "6:\n\t"\
    INIT_SAVE\
    "cmp %w[c_rowmajor],#0; b.eq 7f\n\t"\
    SAVE_M4N##ndim(CR) "b 8f\n\t"\
    "7:\n\t"\
    SAVE_M4N##ndim(CC)\
    "8:\n\t"\
  ::[a_ptr]"r"(a_ptr), [b_scr]"r"(b_scr), [c_ptr]"r"(c_ptr),\
    [K]"r"(K), [LDA]"r"(LDA), [LDC]"r"(LDC),\
    [beta_addr]"r"(beta_addr), [c_rowmajor]"r"(c_rowmajor)\
  :"cc","memory","x0","x1","x2","x3","x4","x5","x6","x7","x8","x9",\
   "x12","x13","x14","x15",\
   "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13",\
   "v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25",\
   "v26","v27","v28","v29","v30","v31");\
}

#define UNIT_SAVE_M3N4_CC(c1, c2, c3) \
  "ldr d1,[x12]; ldr s2,[x12,#8]\n\t"\
  "ldr d3,[x13]; ldr s4,[x13,#8]\n\t"\
  "trn1 v5.4s,v"#c1".4s,v"#c2".4s; trn2 v"#c2".4s,v"#c1".4s,v"#c2".4s\n\t"\
  "mov v6.8b,v"#c2".8b; mov v"#c1".16b,v5.16b\n\t"\
  "fmla v5.2s,v1.2s,v0.s[0]; fmla v6.2s,v3.2s,v0.s[0]\n\t"\
  "fmov s1,s"#c3"; ins v3.s[0],v"#c3".s[1]\n\t"\
  "fmla s1,s2,v0.s[0]; fmla s3,s4,v0.s[0]\n\t"\
  "str d5,[x12]; str s1,[x12,#8]; prfm pstl2keep,[x12,#24]\n\t"\
  "add x12,x12,%w[LDC],UXTW #4\n\t"\
  "str d6,[x13]; str s3,[x13,#8]; prfm pstl2keep,[x13,#24]\n\t"\
  "add x13,x13,%w[LDC],UXTW #4\n\t"\
  "ldr d1,[x14]; ldr s2,[x14,#8]\n\t"\
  "ldr d3,[x15]; ldr s4,[x15,#8]\n\t"\
  "ins v"#c1".d[0],v"#c1".d[1]; ins v"#c2".d[0],v"#c2".d[1]\n\t"\
  "ins v5.s[0],v"#c3".s[2]; ins v6.s[0],v"#c3".s[3]\n\t"\
  "fmla v"#c1".2s,v1.2s,v0.s[0]; fmla v"#c2".2s,v3.2s,v0.s[0]\n\t"\
  "fmla s5,s2,v0.s[0]; fmla s6,s4,v0.s[0]\n\t"\
  "str d"#c1",[x14]; str s5,[x14,#8]; prfm pstl2keep,[x14,#24]\n\t"\
  "add x14,x14,%w[LDC],UXTW #4\n\t"\
  "str d"#c2",[x15]; str s6,[x15,#8]; prfm pstl2keep,[x15,#24]\n\t"\
  "add x15,x15,%w[LDC],UXTW #4\n\t"

#define UNIT_SAVE_M3N4_CR(c1, c2, c3) \
  "ldr q1,[x12]; ldr q2,[x13]; ldr q3,[x14]\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]\n\t"\
  "fmla v"#c2".4s,v2.4s,v0.s[0]\n\t"\
  "fmla v"#c3".4s,v3.4s,v0.s[0]\n\t"\
  "str q"#c1",[x12],#16; str q"#c2",[x13],#16; str q"#c3",[x14],#16\n\t"

#define EDGE_SAVE_M3N1_CC(c1, c2, c3) \
  "ldr d1,[x12]; ldr s2,[x12,#8]\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c2".4s\n\t"\
  "faddp v"#c3".4s,v"#c3".4s,v"#c3".4s\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c1".4s\n\t"\
  "faddp s"#c3",v"#c3".2s\n\t"\
  "fmla v"#c1".2s,v1.2s,v0.s[0]; fmla s"#c3",s2,v0.s[0]\n\t"\
  "str d"#c1",[x12]; str s"#c3",[x12,#8]\n\t"\
  "prfm pstl2keep,[x12,#24]\n\t"\
  "add x12,x12,%w[LDC],UXTW #2\n\t"

#define EDGE_SAVE_M3N1_CR(c1, c2, c3) \
  "ldr s1,[x12]; ldr s2,[x13]; ldr s3,[x14]\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c2".4s\n\t"\
  "faddp v"#c3".4s,v"#c3".4s,v"#c3".4s\n\t"\
  "ins v1.s[1],v2.s[0]\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c1".4s\n\t"\
  "faddp s"#c3",v"#c3".2s\n\t"\
  "fmla v"#c1".2s,v1.2s,v0.s[0]; fmla s"#c3",s3,v0.s[0]\n\t"\
  "st1 {v"#c1".s}[0],[x12],#4; st1 {v"#c1".s}[1],[x13],#4\n\t"\
  "str s"#c3",[x14],#4\n\t"

#define FUNC_M3(ndim) \
static inline void sgemm_skinny1_a7x_m3n##ndim(\
  const float * __restrict__ a_ptr, const float * __restrict__ b_scr,\
  float * __restrict__ c_ptr, uint32_t K, uint32_t LDA, uint32_t LDC,\
  uint8_t c_rowmajor, const float * __restrict__ beta_addr) {\
  __asm__ __volatile__ (\
    "mov x0,%[a_ptr]; add x1,%[a_ptr],%w[LDA],UXTW #2\n\t"\
    "add x2,%[a_ptr],%w[LDA],UXTW #3\n\t"\
    "add x6,x1,%w[LDA],UXTW #3; add x7,x0,%w[LDA],UXTW #4\n\t"\
    "add x8,x1,%w[LDA],UXTW #4\n\t"\
    "mov x4,%[b_scr]; mov w5,%w[K]\n\t"\
    INIT_M3N##ndim\
    "cmp w5,#4; b.lt 4f\n\t"\
    KERNEL_M3N##ndim##_PRELOAD4\
    "cmp w5,#20; b.lt 1f\n\t"\
    ".balign 16; 9:\n\t"\
    KERNEL_M3N##ndim##_MAIN4(0, 1, 2, 3, 4, 5)\
    "prfm pldl2keep,[x6]; add x6,x6,#64\n\t"\
    KERNEL_M3N##ndim##_MAIN4(3, 4, 5, 0, 1, 2)\
    "prfm pldl2keep,[x7]; add x7,x7,#64\n\t"\
    KERNEL_M3N##ndim##_MAIN4(0, 1, 2, 3, 4, 5)\
    "prfm pldl2keep,[x8]; add x8,x8,#64\n\t"\
    KERNEL_M3N##ndim##_MAIN4(3, 4, 5, 0, 1, 2)\
    "cmp w5,#20; b.ge 9b; 1:\n\t"\
    "cmp w5,#12; b.lt 2f\n\t"\
    KERNEL_M3N##ndim##_MAIN4(0, 1, 2, 3, 4, 5)\
    KERNEL_M3N##ndim##_MAIN4(3, 4, 5, 0, 1, 2)\
    "2:\n\t"\
    "cmp w5,#8; b.lt 3f\n\t"\
    KERNEL_M3N##ndim##_MAIN4(0, 1, 2, 3, 4, 5)\
    KERNEL_M3N##ndim##_TAIL4(3, 4, 5)\
    "b 4f; 3:\n\t"\
    KERNEL_M3N##ndim##_TAIL4(0, 1, 2)\
    "4:\n\t"\
    "cmp w5,#1; b.lt 6f\n\t"\
    "5:\n\t"\
    KERNEL_M3N##ndim##_TL1 "b.gt 5b\n\t"\
    "6:\n\t"\
    INIT_SAVE\
    "cmp %w[c_rowmajor],#0; b.eq 7f\n\t"\
    SAVE_M3N##ndim(CR) "b 8f\n\t"\
    "7:\n\t"\
    SAVE_M3N##ndim(CC)\
    "8:\n\t"\
  ::[a_ptr]"r"(a_ptr), [b_scr]"r"(b_scr), [c_ptr]"r"(c_ptr),\
    [K]"r"(K), [LDA]"r"(LDA), [LDC]"r"(LDC),\
    [beta_addr]"r"(beta_addr), [c_rowmajor]"r"(c_rowmajor)\
  :"cc","memory","x0","x1","x2","x4","x5","x6","x7","x8",\
   "x12","x13","x14","x15",\
   "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13",\
   "v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25",\
   "v26","v27","v28","v29","v30","v31");\
}


#define INIT_M4N4 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)

#define SAVE_M4N4(mode) UNIT_SAVE_M4N4_##mode(12, 13, 14, 15)

#define KERNEL_M4N4_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr q3,[x3],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]; ldr q11,[x4,#48]\n\t"\
  "add x4,x4,#64\n\t"

#define KERNEL_M4N4_MAIN4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "ldr q9,[x4,#16]; sub w5,w5,#4\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; fmla v13.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[2]; fmla v15.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#32]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "ldr q"#an4",[x3],#16\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "ldr q11,[x4,#48]; add x4,x4,#64\n\t"

#define KERNEL_M4N4_TAIL4(ac1, ac2, ac3, ac4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; fmla v13.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[2]; fmla v15.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "sub w5,w5,#4; prfm pldl2keep,[x9]\n\t"\
  "fadd v12.4s,v12.4s,v16.4s; fadd v13.4s,v13.4s,v17.4s\n\t"\
  "fadd v14.4s,v14.4s,v18.4s; fadd v15.4s,v15.4s,v19.4s\n\t"

#define KERNEL_M4N4_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "ldr q8,[x4],#16\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v8.4s,v3.s[0]\n\t"


#define INIT_M4N5 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23)

#define SAVE_M4N5(mode) UNIT_SAVE_M4N4_##mode(12, 13, 14, 15)\
  EDGE_SAVE_M4N1_##mode(20, 21, 22, 23)

#define KERNEL_M4N5_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr q3,[x3],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]; ldr q11,[x4,#48]\n\t"\
  "ldr q24,[x4,#64]; add x4,x4,#80\n\t"

#define KERNEL_M4N5_MAIN4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "ldr q9,[x4,#16]; sub w5,w5,#4\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; fmla v13.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[2]; fmla v15.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#32]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "ldr q"#an4",[x3],#16\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "ldr q11,[x4,#48]\n\t"\
  "fmla v20.4s,v24.4s,v"#ac1".4s; fmla v21.4s,v24.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v24.4s,v"#ac3".4s; fmla v23.4s,v24.4s,v"#ac4".4s\n\t"\
  "ldr q24,[x4,#64]; add x4,x4,#80\n\t"

#define KERNEL_M4N5_TAIL4(ac1, ac2, ac3, ac4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; fmla v13.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[2]; fmla v15.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "prfm pldl2keep,[x9]\n\t"\
  "fmla v20.4s,v24.4s,v"#ac1".4s; fmla v21.4s,v24.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v24.4s,v"#ac3".4s; fmla v23.4s,v24.4s,v"#ac4".4s\n\t"\
  "sub w5,w5,#4\n\t"\
  "fadd v12.4s,v12.4s,v16.4s; fadd v13.4s,v13.4s,v17.4s\n\t"\
  "fadd v14.4s,v14.4s,v18.4s; fadd v15.4s,v15.4s,v19.4s\n\t"

#define KERNEL_M4N5_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "ldr q8,[x4],#16; ldr s9,[x4],#4\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v20.4s,v9.4s,v0.4s; fmla v21.4s,v9.4s,v1.4s\n\t"\
  "fmla v22.4s,v9.4s,v2.4s; fmla v23.4s,v9.4s,v3.4s\n\t"


#define INIT_M4N6 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23) INIT_4V(24, 25, 26, 27)

#define SAVE_M4N6(mode) UNIT_SAVE_M4N4_##mode(12, 13, 14, 15)\
  EDGE_SAVE_M4N1_##mode(20, 21, 22, 23) EDGE_SAVE_M4N1_##mode(24, 25, 26, 27)

#define KERNEL_M4N6_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr q3,[x3],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]; ldr q11,[x4,#48]\n\t"\
  "ldr q28,[x4,#64]; ldr q29,[x4,#80]; add x4,x4,#96\n\t"

#define KERNEL_M4N6_MAIN4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "ldr q9,[x4,#16]; sub w5,w5,#4\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; fmla v13.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[2]; fmla v15.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#32]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "ldr q"#an4",[x3],#16\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "ldr q11,[x4,#48]\n\t"\
  "fmla v20.4s,v28.4s,v"#ac1".4s; fmla v21.4s,v28.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v28.4s,v"#ac3".4s; fmla v23.4s,v28.4s,v"#ac4".4s\n\t"\
  "ldr q28,[x4,#64]\n\t"\
  "fmla v24.4s,v29.4s,v"#ac1".4s; fmla v25.4s,v29.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v29.4s,v"#ac3".4s; fmla v27.4s,v29.4s,v"#ac4".4s\n\t"\
  "ldr q29,[x4,#80]; add x4,x4,#96\n\t"

#define KERNEL_M4N6_TAIL4(ac1, ac2, ac3, ac4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; fmla v13.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[2]; fmla v15.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "prfm pldl2keep,[x9]\n\t"\
  "fmla v20.4s,v28.4s,v"#ac1".4s; fmla v21.4s,v28.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v28.4s,v"#ac3".4s; fmla v23.4s,v28.4s,v"#ac4".4s\n\t"\
  "fmla v24.4s,v29.4s,v"#ac1".4s; fmla v25.4s,v29.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v29.4s,v"#ac3".4s; fmla v27.4s,v29.4s,v"#ac4".4s\n\t"\
  "sub w5,w5,#4\n\t"\
  "fadd v12.4s,v12.4s,v16.4s; fadd v13.4s,v13.4s,v17.4s\n\t"\
  "fadd v14.4s,v14.4s,v18.4s; fadd v15.4s,v15.4s,v19.4s\n\t"

#define KERNEL_M4N6_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "ldr q8,[x4],#16; ldr s9,[x4],#4; ldr s10,[x4],#4\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v20.4s,v9.4s,v0.4s; fmla v21.4s,v9.4s,v1.4s\n\t"\
  "fmla v22.4s,v9.4s,v2.4s; fmla v23.4s,v9.4s,v3.4s\n\t"\
  "fmla v24.4s,v10.4s,v0.4s; fmla v25.4s,v10.4s,v1.4s\n\t"\
  "fmla v26.4s,v10.4s,v2.4s; fmla v27.4s,v10.4s,v3.4s\n\t"


#define INIT_M4N7 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23) INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N7(mode) UNIT_SAVE_M4N4_##mode(12, 13, 14, 15)\
  EDGE_SAVE_M4N1_##mode(20, 21, 22, 23) EDGE_SAVE_M4N1_##mode(24, 25, 26, 27)\
  EDGE_SAVE_M4N1_##mode(28, 29, 30, 31)

#define KERNEL_M4N7_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr q3,[x3],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]; ldr q11,[x4,#48]\n\t"\
  "add x4,x4,#112\n\t"

#define KERNEL_M4N7_MAIN4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-48]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "ldr q9,[x4,#-32]; sub w5,w5,#4\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; fmla v13.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[2]; fmla v15.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#-16]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "ldr q"#an4",[x3],#16\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".4s; fmla v21.4s,v8.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".4s; fmla v23.4s,v8.4s,v"#ac4".4s\n\t"\
  "ldr q8,[x4],#112\n\t"\
  "fmla v24.4s,v9.4s,v"#ac1".4s; fmla v25.4s,v9.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v9.4s,v"#ac3".4s; fmla v27.4s,v9.4s,v"#ac4".4s\n\t"\
  "ldr q9,[x4,#-96]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac1".4s; fmla v29.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v30.4s,v10.4s,v"#ac3".4s; fmla v31.4s,v10.4s,v"#ac4".4s\n\t"\
  "ldr q10,[x4,#-80]; ldr q11,[x4,#-64]\n\t"

#define KERNEL_M4N7_TAIL4(ac1, ac2, ac3, ac4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-48]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "ldr q9,[x4,#-32]; sub w5,w5,#4\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; fmla v13.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[2]; fmla v15.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#-16]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "prfm pldl2keep,[x9]\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".4s; fmla v21.4s,v8.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".4s; fmla v23.4s,v8.4s,v"#ac4".4s\n\t"\
  "fmla v24.4s,v9.4s,v"#ac1".4s; fmla v25.4s,v9.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v9.4s,v"#ac3".4s; fmla v27.4s,v9.4s,v"#ac4".4s\n\t"\
  "fmla v28.4s,v10.4s,v"#ac1".4s; fmla v29.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v30.4s,v10.4s,v"#ac3".4s; fmla v31.4s,v10.4s,v"#ac4".4s\n\t"\
  "fadd v12.4s,v12.4s,v16.4s; fadd v13.4s,v13.4s,v17.4s\n\t"\
  "fadd v14.4s,v14.4s,v18.4s; fadd v15.4s,v15.4s,v19.4s\n\t"

#define KERNEL_M4N7_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "ldr q8,[x4],#16; ldr s9,[x4],#4; ldr s10,[x4],#4; ldr s11,[x4],#4\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v20.4s,v9.4s,v0.4s; fmla v21.4s,v9.4s,v1.4s\n\t"\
  "fmla v22.4s,v9.4s,v2.4s; fmla v23.4s,v9.4s,v3.4s\n\t"\
  "fmla v24.4s,v10.4s,v0.4s; fmla v25.4s,v10.4s,v1.4s\n\t"\
  "fmla v26.4s,v10.4s,v2.4s; fmla v27.4s,v10.4s,v3.4s\n\t"\
  "fmla v28.4s,v11.4s,v0.4s; fmla v29.4s,v11.4s,v1.4s\n\t"\
  "fmla v30.4s,v11.4s,v2.4s; fmla v31.4s,v11.4s,v3.4s\n\t"


#define INIT_M4N8 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)

#define SAVE_M4N8(mode) UNIT_SAVE_M4N4_##mode(12, 13, 14, 15)\
  UNIT_SAVE_M4N4_##mode(16, 17, 18, 19)

#define KERNEL_M4N8_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr q3,[x3],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]; ldr q11,[x4,#48]\n\t"\
  "add x4,x4,#128\n\t"

#define KERNEL_M4N8_MAIN4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-64]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-48]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]; fmla v13.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[1]; fmla v15.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-32]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[1]; fmla v17.4s,v11.4s,v"#ac2".s[1]\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[1]; fmla v19.4s,v11.4s,v"#ac4".s[1]\n\t"\
  "ldr q11,[x4,#-16]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[2]; fmla v13.4s,v8.4s,v"#ac2".s[2]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[2]; fmla v15.4s,v8.4s,v"#ac4".s[2]\n\t"\
  "ldr q8,[x4],#128\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[2]; fmla v17.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[2]; fmla v19.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "ldr q9,[x4,#-112]; sub w5,w5,#4\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[3]; fmla v13.4s,v10.4s,v"#ac2".s[3]\n\t"\
  "ldr q"#an4",[x3],#16\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[3]; fmla v15.4s,v10.4s,v"#ac4".s[3]\n\t"\
  "ldr q10,[x4,#-96]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "ldr q11,[x4,#-80]\n\t"

#define KERNEL_M4N8_TAIL4(ac1, ac2, ac3, ac4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-64]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-48]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]; fmla v13.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[1]; fmla v15.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-32]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[1]; fmla v17.4s,v11.4s,v"#ac2".s[1]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[1]; fmla v19.4s,v11.4s,v"#ac4".s[1]\n\t"\
  "ldr q11,[x4,#-16]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[2]; fmla v13.4s,v8.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[2]; fmla v15.4s,v8.4s,v"#ac4".s[2]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[2]; fmla v17.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[2]; fmla v19.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "sub w5,w5,#4\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[3]; fmla v13.4s,v10.4s,v"#ac2".s[3]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[3]; fmla v15.4s,v10.4s,v"#ac4".s[3]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".s[3]; fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "prfm pldl2keep,[x9]\n\t"\
  "fmla v18.4s,v11.4s,v"#ac3".s[3]; fmla v19.4s,v11.4s,v"#ac4".s[3]\n\t"

#define KERNEL_M4N8_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "ldr q8,[x4],#16; ldr q9,[x4],#16\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v16.4s,v9.4s,v0.s[0]; fmla v17.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v9.4s,v2.s[0]; fmla v19.4s,v9.4s,v3.s[0]\n\t"


#define INIT_M4N9 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23)

#define SAVE_M4N9(mode) UNIT_SAVE_M4N4_##mode(12, 13, 14, 15)\
  UNIT_SAVE_M4N4_##mode(16, 17, 18, 19) EDGE_SAVE_M4N1_##mode(20, 21, 22, 23)

#define KERNEL_M4N9_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr q3,[x3],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]\n\t"\
  "add x4,x4,#144\n\t"

#define KERNEL_M4N9_MAIN4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-96]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-80]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]; fmla v13.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[1]; fmla v15.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-64]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]; fmla v17.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac3".s[1]; fmla v19.4s,v8.4s,v"#ac4".s[1]\n\t"\
  "ldr q8,[x4,#-48]\n\t"\
  "fmla v12.4s,v9.4s,v"#ac1".s[2]; fmla v13.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  "fmla v14.4s,v9.4s,v"#ac3".s[2]; fmla v15.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "ldr q9,[x4,#-32]\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[2]; fmla v17.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac3".s[2]; fmla v19.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#-16]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[3]; fmla v13.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "ldr q"#an4",[x3],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[3]; fmla v15.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "ldr q8,[x4],#144\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[3]; fmla v17.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "sub w5,w5,#4\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[3]; fmla v19.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "ldr q9,[x4,#-128]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".4s; fmla v21.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".4s; fmla v23.4s,v10.4s,v"#ac4".4s\n\t"\
  "ldr q10,[x4,#-112]\n\t"

#define KERNEL_M4N9_TAIL4(ac1, ac2, ac3, ac4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-96]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-80]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]; fmla v13.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[1]; fmla v15.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-64]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]; fmla v17.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac3".s[1]; fmla v19.4s,v8.4s,v"#ac4".s[1]\n\t"\
  "ldr q8,[x4,#-48]\n\t"\
  "fmla v12.4s,v9.4s,v"#ac1".s[2]; fmla v13.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v9.4s,v"#ac3".s[2]; fmla v15.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "ldr q9,[x4,#-32]\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[2]; fmla v17.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac3".s[2]; fmla v19.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#-16]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[3]; fmla v13.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[3]; fmla v15.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[3]; fmla v17.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "prfm pldl2keep,[x9]; sub w5,w5,#4\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[3]; fmla v19.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".4s; fmla v21.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".4s; fmla v23.4s,v10.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N9_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "ldr q8,[x4],#16; ldr q9,[x4],#16; ldr s10,[x4],#4\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v16.4s,v9.4s,v0.s[0]; fmla v17.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v9.4s,v2.s[0]; fmla v19.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v20.4s,v10.4s,v0.4s; fmla v21.4s,v10.4s,v1.4s\n\t"\
  "fmla v22.4s,v10.4s,v2.4s; fmla v23.4s,v10.4s,v3.4s\n\t"


#define INIT_M4N10 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23) INIT_4V(24, 25, 26, 27)

#define SAVE_M4N10(mode) UNIT_SAVE_M4N4_##mode(12, 13, 14, 15)\
  UNIT_SAVE_M4N4_##mode(16, 17, 18, 19)\
  EDGE_SAVE_M4N1_##mode(20, 21, 22, 23) EDGE_SAVE_M4N1_##mode(24, 25, 26, 27)

#define KERNEL_M4N10_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr q3,[x3],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]\n\t"\
  "add x4,x4,#160\n\t"

#define KERNEL_M4N10_MAIN4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-112]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-96]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]; fmla v13.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[1]; fmla v15.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-80]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]; fmla v17.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac3".s[1]; fmla v19.4s,v8.4s,v"#ac4".s[1]\n\t"\
  "ldr q8,[x4,#-64]\n\t"\
  "fmla v12.4s,v9.4s,v"#ac1".s[2]; fmla v13.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  "fmla v14.4s,v9.4s,v"#ac3".s[2]; fmla v15.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "ldr q9,[x4,#-48]\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[2]; fmla v17.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac3".s[2]; fmla v19.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#-32]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[3]; fmla v13.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "ldr q"#an4",[x3],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[3]; fmla v15.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "ldr q11,[x4,#-16]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[3]; fmla v17.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "sub w5,w5,#4\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[3]; fmla v19.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "ldr q8,[x4],#160\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".4s; fmla v21.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".4s; fmla v23.4s,v10.4s,v"#ac4".4s\n\t"\
  "ldr q9,[x4,#-144]\n\t"\
  "fmla v24.4s,v11.4s,v"#ac1".4s; fmla v25.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v11.4s,v"#ac3".4s; fmla v27.4s,v11.4s,v"#ac4".4s\n\t"\
  "ldr q10,[x4,#-128]\n\t"

#define KERNEL_M4N10_TAIL4(ac1, ac2, ac3, ac4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-112]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-96]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]; fmla v13.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[1]; fmla v15.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-80]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]; fmla v17.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac3".s[1]; fmla v19.4s,v8.4s,v"#ac4".s[1]\n\t"\
  "ldr q8,[x4,#-64]\n\t"\
  "fmla v12.4s,v9.4s,v"#ac1".s[2]; fmla v13.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v9.4s,v"#ac3".s[2]; fmla v15.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "ldr q9,[x4,#-48]\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[2]; fmla v17.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac3".s[2]; fmla v19.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#-32]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[3]; fmla v13.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[3]; fmla v15.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "ldr q11,[x4,#-16]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[3]; fmla v17.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "prfm pldl2keep,[x9]; sub w5,w5,#4\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[3]; fmla v19.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".4s; fmla v21.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".4s; fmla v23.4s,v10.4s,v"#ac4".4s\n\t"\
  "fmla v24.4s,v11.4s,v"#ac1".4s; fmla v25.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v11.4s,v"#ac3".4s; fmla v27.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N10_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "ldr q8,[x4],#16; ldr q9,[x4],#16; ldr s10,[x4],#4; ldr s11,[x4],#4\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v16.4s,v9.4s,v0.s[0]; fmla v17.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v9.4s,v2.s[0]; fmla v19.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v20.4s,v10.4s,v0.4s; fmla v21.4s,v10.4s,v1.4s\n\t"\
  "fmla v22.4s,v10.4s,v2.4s; fmla v23.4s,v10.4s,v3.4s\n\t"\
  "fmla v24.4s,v11.4s,v0.4s; fmla v25.4s,v11.4s,v1.4s\n\t"\
  "fmla v26.4s,v11.4s,v2.4s; fmla v27.4s,v11.4s,v3.4s\n\t"


#define INIT_M4N11 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23) INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N11(mode) UNIT_SAVE_M4N4_##mode(12, 13, 14, 15)\
  UNIT_SAVE_M4N4_##mode(16, 17, 18, 19)\
  EDGE_SAVE_M4N1_##mode(20, 21, 22, 23) EDGE_SAVE_M4N1_##mode(24, 25, 26, 27)\
  EDGE_SAVE_M4N1_##mode(28, 29, 30, 31)

#define KERNEL_M4N11_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr q3,[x3],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]\n\t"\
  "add x4,x4,#176\n\t"

#define KERNEL_M4N11_MAIN4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-128]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-112]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]; fmla v13.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[1]; fmla v15.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-96]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]; fmla v17.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac3".s[1]; fmla v19.4s,v8.4s,v"#ac4".s[1]\n\t"\
  "ldr q11,[x4,#-80]\n\t"\
  "fmla v12.4s,v9.4s,v"#ac1".s[2]; fmla v13.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  "fmla v14.4s,v9.4s,v"#ac3".s[2]; fmla v15.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "ldr q8,[x4,#-64]\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[2]; fmla v17.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac3".s[2]; fmla v19.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q9,[x4,#-48]\n\t"\
  "fmla v12.4s,v11.4s,v"#ac1".s[3]; fmla v13.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "ldr q"#an4",[x3],#16\n\t"\
  "fmla v14.4s,v11.4s,v"#ac3".s[3]; fmla v15.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "ldr q10,[x4,#-32]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[3]; fmla v17.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "sub w5,w5,#4\n\t"\
  "fmla v18.4s,v8.4s,v"#ac3".s[3]; fmla v19.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "ldr q11,[x4,#-16]\n\t"\
  "fmla v20.4s,v9.4s,v"#ac1".4s; fmla v21.4s,v9.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v9.4s,v"#ac3".4s; fmla v23.4s,v9.4s,v"#ac4".4s\n\t"\
  "ldr q8,[x4],#176\n\t"\
  "fmla v24.4s,v10.4s,v"#ac1".4s; fmla v25.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".4s; fmla v27.4s,v10.4s,v"#ac4".4s\n\t"\
  "ldr q9,[x4,#-160]\n\t"\
  "fmla v28.4s,v11.4s,v"#ac1".4s; fmla v29.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmla v30.4s,v11.4s,v"#ac3".4s; fmla v31.4s,v11.4s,v"#ac4".4s\n\t"\
  "ldr q10,[x4,#-144]\n\t"

#define KERNEL_M4N11_TAIL4(ac1, ac2, ac3, ac4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-128]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-112]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]; fmla v13.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac3".s[1]; fmla v15.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-96]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]; fmla v17.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac3".s[1]; fmla v19.4s,v8.4s,v"#ac4".s[1]\n\t"\
  "ldr q11,[x4,#-80]\n\t"\
  "fmla v12.4s,v9.4s,v"#ac1".s[2]; fmla v13.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v9.4s,v"#ac3".s[2]; fmla v15.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "ldr q8,[x4,#-64]\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[2]; fmla v17.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac3".s[2]; fmla v19.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q9,[x4,#-48]\n\t"\
  "fmla v12.4s,v11.4s,v"#ac1".s[3]; fmla v13.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmla v14.4s,v11.4s,v"#ac3".s[3]; fmla v15.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "ldr q10,[x4,#-32]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[3]; fmla v17.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "prfm pldl2keep,[x9]; sub w5,w5,#4\n\t"\
  "fmla v18.4s,v8.4s,v"#ac3".s[3]; fmla v19.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "ldr q11,[x4,#-16]\n\t"\
  "fmla v20.4s,v9.4s,v"#ac1".4s; fmla v21.4s,v9.4s,v"#ac2".4s\n\t"\
  "fmla v22.4s,v9.4s,v"#ac3".4s; fmla v23.4s,v9.4s,v"#ac4".4s\n\t"\
  "fmla v24.4s,v10.4s,v"#ac1".4s; fmla v25.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".4s; fmla v27.4s,v10.4s,v"#ac4".4s\n\t"\
  "fmla v28.4s,v11.4s,v"#ac1".4s; fmla v29.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmla v30.4s,v11.4s,v"#ac3".4s; fmla v31.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N11_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "ldr q8,[x4],#16; ldr q9,[x4],#16; ldr d10,[x4],#8; ldr s11,[x4],#4\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v16.4s,v9.4s,v0.s[0]; fmla v17.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v9.4s,v2.s[0]; fmla v19.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v20.4s,v0.4s,v10.s[0]; fmla v21.4s,v1.4s,v10.s[0]\n\t"\
  "fmla v22.4s,v2.4s,v10.s[0]; fmla v23.4s,v3.4s,v10.s[0]\n\t"\
  "fmla v24.4s,v0.4s,v10.s[1]; fmla v25.4s,v1.4s,v10.s[1]\n\t"\
  "fmla v26.4s,v2.4s,v10.s[1]; fmla v27.4s,v3.4s,v10.s[1]\n\t"\
  "fmla v28.4s,v0.4s,v11.s[0]; fmla v29.4s,v1.4s,v11.s[0]\n\t"\
  "fmla v30.4s,v2.4s,v11.s[0]; fmla v31.4s,v3.4s,v11.s[0]\n\t"


#define INIT_M4N12 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23)

#define SAVE_M4N12(mode) UNIT_SAVE_M4N4_##mode(12, 13, 14, 15)\
  UNIT_SAVE_M4N4_##mode(16, 17, 18, 19) UNIT_SAVE_M4N4_##mode(20, 21, 22, 23)

#define KERNEL_M4N12_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr q3,[x3],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]\n\t"\
  "add x4,x4,#192\n\t"

#define KERNEL_M4N12_MAIN4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-144]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-128]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[0]; fmla v21.4s,v10.4s,v"#ac2".s[0]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[0]; fmla v23.4s,v10.4s,v"#ac4".s[0]\n\t"\
  "ldr q10,[x4,#-112]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[1]; fmla v13.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[1]; fmla v15.4s,v8.4s,v"#ac4".s[1]\n\t"\
  "ldr q8,[x4,#-96]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "ldr q9,[x4,#-80]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[1]; fmla v21.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[1]; fmla v23.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-64]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[2]; fmla v13.4s,v8.4s,v"#ac2".s[2]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[2]; fmla v15.4s,v8.4s,v"#ac4".s[2]\n\t"\
  "ldr q8,[x4,#-48]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[2]; fmla v17.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[2]; fmla v19.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "ldr q9,[x4,#-32]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[2]; fmla v21.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[2]; fmla v23.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#-16]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[3]; fmla v13.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "ldr q"#an4",[x3],#16\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[3]; fmla v15.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "ldr q8,[x4],#192\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[3]; fmla v17.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "sub w5,w5,#4\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[3]; fmla v19.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "ldr q9,[x4,#-176]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[3]; fmla v21.4s,v10.4s,v"#ac2".s[3]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[3]; fmla v23.4s,v10.4s,v"#ac4".s[3]\n\t"\
  "ldr q10,[x4,#-160]\n\t"

#define KERNEL_M4N12_TAIL4(ac1, ac2, ac3, ac4) \
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; fmla v13.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[0]; fmla v15.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "ldr q8,[x4,#-144]; prfm pldl2keep,[x6]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[0]; fmla v17.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[0]; fmla v19.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "ldr q9,[x4,#-128]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[0]; fmla v21.4s,v10.4s,v"#ac2".s[0]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[0]; fmla v23.4s,v10.4s,v"#ac4".s[0]\n\t"\
  "ldr q10,[x4,#-112]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[1]; fmla v13.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[1]; fmla v15.4s,v8.4s,v"#ac4".s[1]\n\t"\
  "ldr q8,[x4,#-96]; prfm pldl2keep,[x7]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[1]; fmla v17.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[1]; fmla v19.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "ldr q9,[x4,#-80]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[1]; fmla v21.4s,v10.4s,v"#ac2".s[1]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[1]; fmla v23.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "ldr q10,[x4,#-64]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[2]; fmla v13.4s,v8.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[2]; fmla v15.4s,v8.4s,v"#ac4".s[2]\n\t"\
  "ldr q8,[x4,#-48]; prfm pldl2keep,[x8]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[2]; fmla v17.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[2]; fmla v19.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "ldr q9,[x4,#-32]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[2]; fmla v21.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[2]; fmla v23.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "ldr q10,[x4,#-16]\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[3]; fmla v13.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v14.4s,v8.4s,v"#ac3".s[3]; fmla v15.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[3]; fmla v17.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "sub w5,w5,#4; prfm pldl2keep,[x9]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac3".s[3]; fmla v19.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[3]; fmla v21.4s,v10.4s,v"#ac2".s[3]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[3]; fmla v23.4s,v10.4s,v"#ac4".s[3]\n\t"

#define KERNEL_M4N12_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "ldr q8,[x4],#16; ldr q9,[x4],#16; ldr q10,[x4],#16\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v16.4s,v9.4s,v0.s[0]; fmla v17.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v9.4s,v2.s[0]; fmla v19.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v20.4s,v10.4s,v0.s[0]; fmla v21.4s,v10.4s,v1.s[0]\n\t"\
  "fmla v22.4s,v10.4s,v2.s[0]; fmla v23.4s,v10.4s,v3.s[0]\n\t"

FUNC_M4(4)
FUNC_M4(5)
FUNC_M4(6)
FUNC_M4(7)
FUNC_M4(8)
FUNC_M4(9)
FUNC_M4(10)
FUNC_M4(11)
FUNC_M4(12)

#define FMA_M3N4(c1, c2, c3, a1, a2, a3, b1, k) \
  "fmla v"#c1".4s,v"#b1".4s,v"#a1".s["#k"]\n\t"\
  "fmla v"#c2".4s,v"#b1".4s,v"#a2".s["#k"]\n\t"\
  "fmla v"#c3".4s,v"#b1".4s,v"#a3".s["#k"]\n\t"

#define FMA_M3N1(c1, c2, c3, a1, a2, a3, b1) \
  "fmla v"#c1".4s,v"#b1".4s,v"#a1".4s\n\t"\
  "fmla v"#c2".4s,v"#b1".4s,v"#a2".4s\n\t"\
  "fmla v"#c3".4s,v"#b1".4s,v"#a3".4s\n\t"


#define INIT_M3N13 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23)

#define SAVE_M3N13(mode) UNIT_SAVE_M3N4_##mode(12, 13, 14)\
  UNIT_SAVE_M3N4_##mode(15, 16, 17) UNIT_SAVE_M3N4_##mode(18, 19, 20)\
  EDGE_SAVE_M3N1_##mode(21, 22, 23)

#define KERNEL_M3N13_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]\n\t"\
  "add x4,x4,#208\n\t"

#define KERNEL_M3N13_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-160]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 0) "ldr q9,[x4,#-144]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 0) "ldr q10,[x4,#-128]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-112]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 1) "ldr q9,[x4,#-96]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 1) "ldr q10,[x4,#-80]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 2) "ldr q8,[x4,#-64]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 2) "ldr q9,[x4,#-48]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 2) "ldr q10,[x4,#-32]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 3) "ldr q11,[x4,#-16]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 3) "ldr q8,[x4],#208\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 3) "ldr q9,[x4,#-192]\n\t"\
  FMA_M3N1(21, 22, 23, ac1, ac2, ac3, 11) "ldr q10,[x4,#-176]\n\t"
 
#define KERNEL_M3N13_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-160]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 0) "ldr q9,[x4,#-144]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 0) "ldr q10,[x4,#-128]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-112]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 1) "ldr q9,[x4,#-96]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 1) "ldr q10,[x4,#-80]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 2) "ldr q8,[x4,#-64]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 2) "ldr q9,[x4,#-48]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 2) "ldr q10,[x4,#-32]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 3) "ldr q11,[x4,#-16]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 3)\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 3)\
  FMA_M3N1(21, 22, 23, ac1, ac2, ac3, 11)

#define KERNEL_M3N13_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q8,[x4],#16; ldr q9,[x4],#16; ldr q10,[x4],#16; ldr s11,[x4],#4\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v16.4s,v9.4s,v1.s[0]; fmla v17.4s,v9.4s,v2.s[0]\n\t"\
  "fmla v18.4s,v10.4s,v0.s[0]; fmla v19.4s,v10.4s,v1.s[0]\n\t"\
  "fmla v20.4s,v10.4s,v2.s[0]; fmla v21.4s,v0.4s,v11.s[0]\n\t"\
  "fmla v22.4s,v1.4s,v11.s[0]; fmla v23.4s,v2.4s,v11.s[0]\n\t"


#define INIT_M3N14 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23) INIT_2V(24, 25) INIT_1V(26)

#define SAVE_M3N14(mode) UNIT_SAVE_M3N4_##mode(12, 13, 14)\
  UNIT_SAVE_M3N4_##mode(15, 16, 17) UNIT_SAVE_M3N4_##mode(18, 19, 20)\
  EDGE_SAVE_M3N1_##mode(21, 22, 23) EDGE_SAVE_M3N1_##mode(24, 25, 26)

#define KERNEL_M3N14_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]\n\t"\
  "add x4,x4,#224\n\t"

#define KERNEL_M3N14_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-176]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 0) "ldr q9,[x4,#-160]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 0) "ldr q10,[x4,#-144]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-128]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 1) "ldr q9,[x4,#-112]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 1) "ldr q10,[x4,#-96]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 2) "ldr q11,[x4,#-80]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 2) "ldr q8,[x4,#-64]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 2) "ldr q9,[x4,#-48]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 11, 3) "ldr q10,[x4,#-32]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 8, 3) "ldr q11,[x4,#-16]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 9, 3) "ldr q8,[x4],#224\n\t"\
  FMA_M3N1(21, 22, 23, ac1, ac2, ac3, 10) "ldr q9,[x4,#-208]\n\t"\
  FMA_M3N1(24, 25, 26, ac1, ac2, ac3, 11) "ldr q10,[x4,#-192]\n\t"

#define KERNEL_M3N14_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-176]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 0) "ldr q9,[x4,#-160]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 0) "ldr q10,[x4,#-144]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-128]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 1) "ldr q9,[x4,#-112]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 1) "ldr q10,[x4,#-96]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 2) "ldr q11,[x4,#-80]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 2) "ldr q8,[x4,#-64]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 2) "ldr q9,[x4,#-48]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 11, 3) "ldr q10,[x4,#-32]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 8, 3) "ldr q11,[x4,#-16]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 9, 3)\
  FMA_M3N1(21, 22, 23, ac1, ac2, ac3, 10)\
  FMA_M3N1(24, 25, 26, ac1, ac2, ac3, 11)

#define KERNEL_M3N14_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q8,[x4],#16; ldr q9,[x4],#16; ldr q10,[x4],#16; ldr d11,[x4],#8\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v16.4s,v9.4s,v1.s[0]; fmla v17.4s,v9.4s,v2.s[0]\n\t"\
  "fmla v18.4s,v10.4s,v0.s[0]; fmla v19.4s,v10.4s,v1.s[0]\n\t"\
  "fmla v20.4s,v10.4s,v2.s[0]; fmla v21.4s,v0.4s,v11.s[0]\n\t"\
  "fmla v22.4s,v1.4s,v11.s[0]; fmla v23.4s,v2.4s,v11.s[0]\n\t"\
  "fmla v24.4s,v0.4s,v11.s[1]; fmla v25.4s,v1.4s,v11.s[1]\n\t"\
  "fmla v26.4s,v2.4s,v11.s[1]\n\t"


#define INIT_M3N15 INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23) INIT_4V(24, 25, 26, 27) INIT_2V(28, 29)

#define SAVE_M3N15(mode) UNIT_SAVE_M3N4_##mode(12, 13, 14)\
  UNIT_SAVE_M3N4_##mode(15, 16, 17) UNIT_SAVE_M3N4_##mode(18, 19, 20)\
  EDGE_SAVE_M3N1_##mode(21, 22, 23) EDGE_SAVE_M3N1_##mode(24, 25, 26)\
  EDGE_SAVE_M3N1_##mode(27, 28, 29)

#define KERNEL_M3N15_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; ldr q10,[x4,#32]\n\t"\
  "add x4,x4,#240\n\t"

#define KERNEL_M3N15_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-192]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 0) "ldr q9,[x4,#-176]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 0) "ldr q10,[x4,#-160]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 1) "ldr q11,[x4,#-144]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 1) "ldr q8,[x4,#-128]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 1) "ldr q9,[x4,#-112]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 11, 2) "ldr q10,[x4,#-96]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 8, 2) "ldr q11,[x4,#-80]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 9, 2) "ldr q8,[x4,#-64]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 10, 3) "ldr q9,[x4,#-48]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 11, 3) "ldr q10,[x4,#-32]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 8, 3) "ldr q11,[x4,#-16]\n\t"\
  FMA_M3N1(21, 22, 23, ac1, ac2, ac3, 9) "ldr q8,[x4],#240\n\t"\
  FMA_M3N1(24, 25, 26, ac1, ac2, ac3, 10) "ldr q9,[x4,#-224]\n\t"\
  FMA_M3N1(27, 28, 29, ac1, ac2, ac3, 11) "ldr q10,[x4,#-208]\n\t"

#define KERNEL_M3N15_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-192]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 0) "ldr q9,[x4,#-176]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 0) "ldr q10,[x4,#-160]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 8, 1) "ldr q11,[x4,#-144]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 9, 1) "ldr q8,[x4,#-128]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 10, 1) "ldr q9,[x4,#-112]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 11, 2) "ldr q10,[x4,#-96]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 8, 2) "ldr q11,[x4,#-80]\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 9, 2) "ldr q8,[x4,#-64]\n\t"\
  FMA_M3N4(12, 13, 14, ac1, ac2, ac3, 10, 3) "ldr q9,[x4,#-48]\n\t"\
  FMA_M3N4(15, 16, 17, ac1, ac2, ac3, 11, 3) "ldr q10,[x4,#-32]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(18, 19, 20, ac1, ac2, ac3, 8, 3) "ldr q11,[x4,#-16]\n\t"\
  FMA_M3N1(21, 22, 23, ac1, ac2, ac3, 9)\
  FMA_M3N1(24, 25, 26, ac1, ac2, ac3, 10)\
  FMA_M3N1(27, 28, 29, ac1, ac2, ac3, 11)

#define KERNEL_M3N15_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q8,[x4],#16; ldr q9,[x4],#16; ldr q10,[x4],#16; ldr d11,[x4],#8\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]; fmla v13.4s,v8.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v14.4s,v8.4s,v2.s[0]; fmla v15.4s,v9.4s,v0.s[0]\n\t"\
  "ldr s8,[x4],#4\n\t"\
  "fmla v16.4s,v9.4s,v1.s[0]; fmla v17.4s,v9.4s,v2.s[0]\n\t"\
  "fmla v18.4s,v10.4s,v0.s[0]; fmla v19.4s,v10.4s,v1.s[0]\n\t"\
  "fmla v20.4s,v10.4s,v2.s[0]; fmla v21.4s,v0.4s,v11.s[0]\n\t"\
  "fmla v22.4s,v1.4s,v11.s[0]; fmla v23.4s,v2.4s,v11.s[0]\n\t"\
  "fmla v24.4s,v0.4s,v11.s[1]; fmla v25.4s,v1.4s,v11.s[1]\n\t"\
  "fmla v26.4s,v2.4s,v11.s[1]; fmla v27.4s,v0.4s,v8.s[0]\n\t"\
  "fmla v28.4s,v1.4s,v8.s[0]; fmla v29.4s,v2.4s,v8.s[0]\n\t"


#define INIT_M3N16 INIT_4V(10, 11, 12, 13) INIT_4V(14, 15, 16, 17)\
  INIT_4V(18, 19, 20, 21)

#define SAVE_M3N16(mode) UNIT_SAVE_M3N4_##mode(10, 11, 12)\
  UNIT_SAVE_M3N4_##mode(13, 14, 15) UNIT_SAVE_M3N4_##mode(16, 17, 18)\
  UNIT_SAVE_M3N4_##mode(19, 20, 21)

#define KERNEL_M3N16_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]; ldr q8,[x4,#32]\n\t"\
  "add x4,x4,#256\n\t"

#define KERNEL_M3N16_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-208]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#-192]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-176]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-144]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-128]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-112]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-96]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 8, 2) "ldr q8,[x4,#-80]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 8, 2) "ldr q8,[x4,#-32]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 3) "ldr q9,[x4,#-16]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 3) "ldr q6,[x4]; add x4,x4,#256\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 3) "ldr q7,[x4,#-240]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 9, 3) "ldr q8,[x4,#-224]\n\t"

#define KERNEL_M3N16_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-208]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#-192]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-176]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-144]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-128]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-112]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-96]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 8, 2) "ldr q8,[x4,#-80]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 8, 2) "ldr q8,[x4,#-32]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 3) "ldr q9,[x4,#-16]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 3)\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 3)\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 9, 3)

#define KERNEL_M3N16_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16; ldr q8,[x4],#16; ldr q9,[x4],#16\n\t"\
  "fmla v10.4s,v6.4s,v0.s[0]; fmla v11.4s,v6.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v12.4s,v6.4s,v2.s[0]; fmla v13.4s,v7.4s,v0.s[0]\n\t"\
  "fmla v14.4s,v7.4s,v1.s[0]; fmla v15.4s,v7.4s,v2.s[0]\n\t"\
  "fmla v16.4s,v8.4s,v0.s[0]; fmla v17.4s,v8.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v8.4s,v2.s[0]; fmla v19.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v20.4s,v9.4s,v1.s[0]; fmla v21.4s,v9.4s,v2.s[0]\n\t"


#define INIT_M3N17 INIT_4V(10, 11, 12, 13) INIT_4V(14, 15, 16, 17)\
  INIT_4V(18, 19, 20, 21) INIT_2V(22, 23) INIT_1V(24)

#define SAVE_M3N17(mode) UNIT_SAVE_M3N4_##mode(10, 11, 12)\
  UNIT_SAVE_M3N4_##mode(13, 14, 15) UNIT_SAVE_M3N4_##mode(16, 17, 18)\
  UNIT_SAVE_M3N4_##mode(19, 20, 21) EDGE_SAVE_M3N1_##mode(22, 23, 24)

#define KERNEL_M3N17_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]; ldr q8,[x4,#32]\n\t"\
  "add x4,x4,#272\n\t"

#define KERNEL_M3N17_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-224]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#-208]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-192]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-176]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-160]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-144]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 8, 2) "ldr q8,[x4,#-96]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 6, 2) "ldr q9,[x4,#-80]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 7, 2) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 8, 2) "ldr q7,[x4,#-48]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 9, 3) "ldr q8,[x4,#-32]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 6, 3) "ldr q9,[x4,#-16]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 7, 3) "ldr q6,[x4]; add x4,x4,#272\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 8, 3) "ldr q7,[x4,#-256]\n\t"\
  FMA_M3N1(22, 23, 24, ac1, ac2, ac3, 9) "ldr q8,[x4,#-240]\n\t"

#define KERNEL_M3N17_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-224]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#-208]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-192]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-176]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-160]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-144]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 8, 2) "ldr q8,[x4,#-96]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 6, 2) "ldr q9,[x4,#-80]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 7, 2) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 8, 2) "ldr q7,[x4,#-48]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 9, 3) "ldr q8,[x4,#-32]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 6, 3) "ldr q9,[x4,#-16]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 7, 3)\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 8, 3)\
  FMA_M3N1(22, 23, 24, ac1, ac2, ac3, 9)

#define KERNEL_M3N17_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16; ldr q8,[x4],#16; ldr q9,[x4],#16\n\t"\
  "fmla v10.4s,v6.4s,v0.s[0]; fmla v11.4s,v6.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v12.4s,v6.4s,v2.s[0]; fmla v13.4s,v7.4s,v0.s[0]; ldr s6,[x4],#4\n\t"\
  "fmla v14.4s,v7.4s,v1.s[0]; fmla v15.4s,v7.4s,v2.s[0]\n\t"\
  "fmla v16.4s,v8.4s,v0.s[0]; fmla v17.4s,v8.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v8.4s,v2.s[0]; fmla v19.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v20.4s,v9.4s,v1.s[0]; fmla v21.4s,v9.4s,v2.s[0]\n\t"\
  "fmla v22.4s,v0.4s,v6.s[0]; fmla v23.4s,v1.4s,v6.s[0]\n\t"\
  "fmla v24.4s,v2.4s,v6.s[0]\n\t"


#define INIT_M3N18 INIT_4V(10, 11, 12, 13) INIT_4V(14, 15, 16, 17)\
  INIT_4V(18, 19, 20, 21) INIT_4V(22, 23, 24, 25) INIT_2V(26, 27)

#define SAVE_M3N18(mode) UNIT_SAVE_M3N4_##mode(10, 11, 12)\
  UNIT_SAVE_M3N4_##mode(13, 14, 15) UNIT_SAVE_M3N4_##mode(16, 17, 18)\
  UNIT_SAVE_M3N4_##mode(19, 20, 21) EDGE_SAVE_M3N1_##mode(22, 23, 24)\
  EDGE_SAVE_M3N1_##mode(25, 26, 27)

#define KERNEL_M3N18_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]; ldr q8,[x4,#32]\n\t"\
  "add x4,x4,#288\n\t"

#define KERNEL_M3N18_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-240]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#-224]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-208]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-176]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-160]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 1) "ldr q9,[x4,#-144]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 1) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 8, 2) "ldr q7,[x4,#-112]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 9, 2) "ldr q8,[x4,#-96]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 2) "ldr q9,[x4,#-80]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 2) "ldr q6,[x4,#-64]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 8, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 9, 3) "ldr q8,[x4,#-32]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 3) "ldr q9,[x4,#-16]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 3) "ldr q6,[x4]\n\t"\
  FMA_M3N1(22, 23, 24, ac1, ac2, ac3, 8) "ldr q7,[x4,#16]\n\t"\
  FMA_M3N1(25, 26, 27, ac1, ac2, ac3, 9) "ldr q8,[x4,#32]; add x4,x4,#288\n\t"

#define KERNEL_M3N18_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-240]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#-224]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-208]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-176]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 1) "ldr q8,[x4,#-160]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 1) "ldr q9,[x4,#-144]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 1) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 8, 2) "ldr q7,[x4,#-112]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 9, 2) "ldr q8,[x4,#-96]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 2) "ldr q9,[x4,#-80]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 2) "ldr q6,[x4,#-64]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 8, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 9, 3) "ldr q8,[x4,#-32]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 6, 3) "ldr q9,[x4,#-16]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 7, 3)\
  FMA_M3N1(22, 23, 24, ac1, ac2, ac3, 8)\
  FMA_M3N1(25, 26, 27, ac1, ac2, ac3, 9)

#define KERNEL_M3N18_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16; ldr q8,[x4],#16; ldr q9,[x4],#16\n\t"\
  "fmla v10.4s,v6.4s,v0.s[0]; fmla v11.4s,v6.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v12.4s,v6.4s,v2.s[0]; fmla v13.4s,v7.4s,v0.s[0]; ldr d6,[x4],#8\n\t"\
  "fmla v14.4s,v7.4s,v1.s[0]; fmla v15.4s,v7.4s,v2.s[0]\n\t"\
  "fmla v16.4s,v8.4s,v0.s[0]; fmla v17.4s,v8.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v8.4s,v2.s[0]; fmla v19.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v20.4s,v9.4s,v1.s[0]; fmla v21.4s,v9.4s,v2.s[0]\n\t"\
  "fmla v22.4s,v0.4s,v6.s[0]; fmla v23.4s,v1.4s,v6.s[0]\n\t"\
  "fmla v24.4s,v2.4s,v6.s[0]; fmla v25.4s,v0.4s,v6.s[1]\n\t"\
  "fmla v26.4s,v1.4s,v6.s[1]; fmla v27.4s,v2.4s,v6.s[1]\n\t"


#define INIT_M3N19 INIT_4V(10, 11, 12, 13) INIT_4V(14, 15, 16, 17)\
  INIT_4V(18, 19, 20, 21) INIT_4V(22, 23, 24, 25)\
  INIT_4V(26, 27, 28, 29) INIT_1V(30)

#define SAVE_M3N19(mode) UNIT_SAVE_M3N4_##mode(10, 11, 12)\
  UNIT_SAVE_M3N4_##mode(13, 14, 15) UNIT_SAVE_M3N4_##mode(16, 17, 18)\
  UNIT_SAVE_M3N4_##mode(19, 20, 21) EDGE_SAVE_M3N1_##mode(22, 23, 24)\
  EDGE_SAVE_M3N1_##mode(25, 26, 27) EDGE_SAVE_M3N1_##mode(28, 29, 30)

#define KERNEL_M3N19_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]; ldr q8,[x4,#32]\n\t"\
  "add x4,x4,#304\n\t"

#define KERNEL_M3N19_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-256]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#-240]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-224]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 0) "ldr q9,[x4,#-208]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 1) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 1) "ldr q7,[x4,#-176]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 9, 1) "ldr q8,[x4,#-160]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 1) "ldr q9,[x4,#-144]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 2) "ldr q6,[x4,#-128]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 2) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 9, 2) "ldr q8,[x4,#-96]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 2) "ldr q9,[x4,#-80]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 3) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 9, 3) "ldr q8,[x4,#-32]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 3) "ldr q9,[x4,#-16]\n\t"\
  FMA_M3N1(22, 23, 24, ac1, ac2, ac3, 7) "ldr q6,[x4]\n\t"\
  FMA_M3N1(25, 26, 27, ac1, ac2, ac3, 8) "ldr q7,[x4,#16]\n\t"\
  FMA_M3N1(28, 29, 30, ac1, ac2, ac3, 9) "ldr q8,[x4,#32]; add x4,x4,#304\n\t"

#define KERNEL_M3N19_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#-256]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#-240]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 8, 0) "ldr q8,[x4,#-224]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 0) "ldr q9,[x4,#-208]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 1) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 1) "ldr q7,[x4,#-176]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 9, 1) "ldr q8,[x4,#-160]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 1) "ldr q9,[x4,#-144]\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 2) "ldr q6,[x4,#-128]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 2) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 9, 2) "ldr q8,[x4,#-96]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 2) "ldr q9,[x4,#-80]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(10, 11, 12, ac1, ac2, ac3, 7, 3) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(13, 14, 15, ac1, ac2, ac3, 8, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(16, 17, 18, ac1, ac2, ac3, 9, 3) "ldr q8,[x4,#-32]\n\t"\
  FMA_M3N4(19, 20, 21, ac1, ac2, ac3, 6, 3) "ldr q9,[x4,#-16]\n\t"\
  FMA_M3N1(22, 23, 24, ac1, ac2, ac3, 7)\
  FMA_M3N1(25, 26, 27, ac1, ac2, ac3, 8)\
  FMA_M3N1(28, 29, 30, ac1, ac2, ac3, 9)

#define KERNEL_M3N19_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16; ldr q8,[x4],#16; ldr q9,[x4],#16\n\t"\
  "fmla v10.4s,v6.4s,v0.s[0]; fmla v11.4s,v6.4s,v1.s[0]; subs w5,w5,#1\n\t"\
  "fmla v12.4s,v6.4s,v2.s[0]; fmla v13.4s,v7.4s,v0.s[0]; ldr d6,[x4],#8\n\t"\
  "fmla v14.4s,v7.4s,v1.s[0]; fmla v15.4s,v7.4s,v2.s[0]; ldr s7,[x4],#4\n\t"\
  "fmla v16.4s,v8.4s,v0.s[0]; fmla v17.4s,v8.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v8.4s,v2.s[0]; fmla v19.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v20.4s,v9.4s,v1.s[0]; fmla v21.4s,v9.4s,v2.s[0]\n\t"\
  "fmla v22.4s,v0.4s,v6.s[0]; fmla v23.4s,v1.4s,v6.s[0]\n\t"\
  "fmla v24.4s,v2.4s,v6.s[0]; fmla v25.4s,v0.4s,v6.s[1]\n\t"\
  "fmla v26.4s,v1.4s,v6.s[1]; fmla v27.4s,v2.4s,v6.s[1]\n\t"\
  "fmla v28.4s,v0.4s,v7.s[0]; fmla v29.4s,v1.4s,v7.s[0]\n\t"\
  "fmla v30.4s,v2.4s,v7.s[0]\n\t"


#define INIT_M3N20 INIT_4V(8, 9, 10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_2V(20, 21) INIT_1V(22)

#define SAVE_M3N20(mode) UNIT_SAVE_M3N4_##mode(8, 9, 10)\
  UNIT_SAVE_M3N4_##mode(11, 12, 13) UNIT_SAVE_M3N4_##mode(14, 15, 16)\
  UNIT_SAVE_M3N4_##mode(17, 18, 19) UNIT_SAVE_M3N4_##mode(20, 21, 22)

#define KERNEL_M3N20_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N20_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  "add x4,x4,#320\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-208]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-176]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-144]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-112]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-96]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-80]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-48]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-32]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 3) "ldr q6,[x4]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N20_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  "add x4,x4,#320\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-208]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-176]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-144]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-112]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-96]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-80]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-48]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-32]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 3)\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 3)

#define KERNEL_M3N20_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16\n\t"\
  FMA_M3N4(8, 9, 10, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(11, 12, 13, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(14, 15, 16, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(17, 18, 19, 0, 1, 2, 7, 0) "subs w5,w5,#1\n\t"\
  FMA_M3N4(20, 21, 22, 0, 1, 2, 6, 0)


#define INIT_M3N21 INIT_4V(8, 9, 10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23) INIT_2V(24, 25)

#define SAVE_M3N21(mode) UNIT_SAVE_M3N4_##mode(8, 9, 10)\
  UNIT_SAVE_M3N4_##mode(11, 12, 13) UNIT_SAVE_M3N4_##mode(14, 15, 16)\
  UNIT_SAVE_M3N4_##mode(17, 18, 19) UNIT_SAVE_M3N4_##mode(20, 21, 22)\
  EDGE_SAVE_M3N1_##mode(23, 24, 25)

#define KERNEL_M3N21_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N21_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  "add x4,x4,#336\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-224]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-208]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-192]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-176]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-160]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-144]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-128]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-112]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-96]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-80]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-64]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-32]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 3) "ldr q6,[x4]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(23, 24, 25, ac1, ac2, ac3, 7) "ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N21_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  "add x4,x4,#336\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-224]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-208]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-192]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-176]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-160]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-144]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-128]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-112]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-96]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-80]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-64]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-32]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 3)\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(23, 24, 25, ac1, ac2, ac3, 7)

#define KERNEL_M3N21_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16\n\t"\
  FMA_M3N4(8, 9, 10, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(11, 12, 13, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(14, 15, 16, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(17, 18, 19, 0, 1, 2, 7, 0) "ldr s7,[x4],#4\n\t"\
  FMA_M3N4(20, 21, 22, 0, 1, 2, 6, 0) "subs w5,w5,#1\n\t"\
  "fmla v23.4s,v0.4s,v7.s[0]; fmla v24.4s,v1.4s,v7.s[0]\n\t"\
  "fmla v25.4s,v2.4s,v7.s[0]\n\t"


#define INIT_M3N22 INIT_4V(8, 9, 10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_1V(28)

#define SAVE_M3N22(mode) UNIT_SAVE_M3N4_##mode(8, 9, 10)\
  UNIT_SAVE_M3N4_##mode(11, 12, 13) UNIT_SAVE_M3N4_##mode(14, 15, 16)\
  UNIT_SAVE_M3N4_##mode(17, 18, 19) UNIT_SAVE_M3N4_##mode(20, 21, 22)\
  EDGE_SAVE_M3N1_##mode(23, 24, 25) EDGE_SAVE_M3N1_##mode(26, 27, 28)\

#define KERNEL_M3N22_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N22_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  "add x4,x4,#352\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-240]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-224]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-208]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-176]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-144]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-96]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-80]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-32]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(23, 24, 25, ac1, ac2, ac3, 6) "ldr q6,[x4]\n\t"\
  FMA_M3N1(26, 27, 28, ac1, ac2, ac3, 7) "ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N22_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  "add x4,x4,#352\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-240]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-224]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-208]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-176]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-144]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-96]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-80]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-32]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(23, 24, 25, ac1, ac2, ac3, 6)\
  FMA_M3N1(26, 27, 28, ac1, ac2, ac3, 7)

#define KERNEL_M3N22_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16\n\t"\
  FMA_M3N4(8, 9, 10, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(11, 12, 13, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(14, 15, 16, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(17, 18, 19, 0, 1, 2, 7, 0) "ldr d7,[x4],#8\n\t"\
  FMA_M3N4(20, 21, 22, 0, 1, 2, 6, 0) "subs w5,w5,#1\n\t"\
  "fmla v23.4s,v0.4s,v7.s[0]; fmla v24.4s,v1.4s,v7.s[0]\n\t"\
  "fmla v25.4s,v2.4s,v7.s[0]; fmla v26.4s,v0.4s,v7.s[1]\n\t"\
  "fmla v27.4s,v1.4s,v7.s[1]; fmla v28.4s,v2.4s,v7.s[1]\n\t"


#define INIT_M3N23 INIT_4V(8, 9, 10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M3N23(mode) UNIT_SAVE_M3N4_##mode(8, 9, 10)\
  UNIT_SAVE_M3N4_##mode(11, 12, 13) UNIT_SAVE_M3N4_##mode(14, 15, 16)\
  UNIT_SAVE_M3N4_##mode(17, 18, 19) UNIT_SAVE_M3N4_##mode(20, 21, 22)\
  EDGE_SAVE_M3N1_##mode(23, 24, 25) EDGE_SAVE_M3N1_##mode(26, 27, 28)\
  EDGE_SAVE_M3N1_##mode(29, 30, 31)

#define KERNEL_M3N23_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N23_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  "add x4,x4,#368\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-256]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-240]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-224]\n\t"\
  "ldr q"#an2",[x1],#16\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-208]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-192]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-176]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-160]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-144]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-128]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-96]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-80]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-48]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-32]\n\t"\
  FMA_M3N1(23, 24, 25, ac1, ac2, ac3, 6) "ldr q6,[x4]\n\t"\
  FMA_M3N1(26, 27, 28, ac1, ac2, ac3, 7) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(29, 30, 31, ac1, ac2, ac3, 7) "ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N23_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  "add x4,x4,#368\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-256]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-240]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-224]\n\t"\
  "prfm pldl2keep,[x7]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-208]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-192]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-176]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-160]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-144]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-128]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-96]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-80]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-48]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-32]\n\t"\
  FMA_M3N1(23, 24, 25, ac1, ac2, ac3, 6)\
  FMA_M3N1(26, 27, 28, ac1, ac2, ac3, 7) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(29, 30, 31, ac1, ac2, ac3, 7)

#define KERNEL_M3N23_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16\n\t"\
  FMA_M3N4(8, 9, 10, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(11, 12, 13, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(14, 15, 16, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(17, 18, 19, 0, 1, 2, 7, 0) "ldr d7,[x4],#8\n\t"\
  FMA_M3N4(20, 21, 22, 0, 1, 2, 6, 0) "ldr s6,[x4],#4\n\t"\
  "fmla v23.4s,v0.4s,v7.s[0]; fmla v24.4s,v1.4s,v7.s[0]; subs w5,w5,#1\n\t"\
  "fmla v25.4s,v2.4s,v7.s[0]; fmla v26.4s,v0.4s,v7.s[1]\n\t"\
  "fmla v27.4s,v1.4s,v7.s[1]; fmla v28.4s,v2.4s,v7.s[1]\n\t"\
  "fmla v29.4s,v0.4s,v6.s[0]; fmla v30.4s,v1.4s,v6.s[0]\n\t"\
  "fmla v31.4s,v2.4s,v6.s[0]\n\t"


#define INIT_M3N24 INIT_4V(8, 9, 10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23) INIT_2V(24, 25)

#define SAVE_M3N24(mode) UNIT_SAVE_M3N4_##mode(8, 9, 10)\
  UNIT_SAVE_M3N4_##mode(11, 12, 13) UNIT_SAVE_M3N4_##mode(14, 15, 16)\
  UNIT_SAVE_M3N4_##mode(17, 18, 19) UNIT_SAVE_M3N4_##mode(20, 21, 22)\
  UNIT_SAVE_M3N4_##mode(23, 24, 25)

#define KERNEL_M3N24_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N24_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#128]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#144]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#160]\n\t"\
  "ldr q"#an2",[x1],#16; add x4,x4,#384\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-208]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-176]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-144]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-96]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-80]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-64]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-32]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 3) "ldr q6,[x4]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N24_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#128]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#144]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#160]\n\t"\
  "prfm pldl2keep,[x7]; add x4,x4,#384\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-208]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-176]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-144]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-96]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-80]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-64]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-32]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 3)\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 3)

#define KERNEL_M3N24_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16\n\t"\
  FMA_M3N4(8, 9, 10, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(11, 12, 13, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(14, 15, 16, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(17, 18, 19, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(20, 21, 22, 0, 1, 2, 6, 0) "subs w5,w5,#1\n\t"\
  FMA_M3N4(23, 24, 25, 0, 1, 2, 7, 0)


#define INIT_M3N25 INIT_4V(8, 9, 10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_1V(28)

#define SAVE_M3N25(mode) UNIT_SAVE_M3N4_##mode(8, 9, 10)\
  UNIT_SAVE_M3N4_##mode(11, 12, 13) UNIT_SAVE_M3N4_##mode(14, 15, 16)\
  UNIT_SAVE_M3N4_##mode(17, 18, 19) UNIT_SAVE_M3N4_##mode(20, 21, 22)\
  UNIT_SAVE_M3N4_##mode(23, 24, 25) EDGE_SAVE_M3N1_##mode(26, 27, 28)

#define KERNEL_M3N25_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N25_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#128]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#144]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#160]\n\t"\
  "ldr q"#an2",[x1],#16; add x4,x4,#400\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-224]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-208]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-192]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-176]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-160]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-144]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-128]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-112]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-96]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-80]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-64]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-48]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-32]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 3) "ldr q6,[x4]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(26, 27, 28, ac1, ac2, ac3, 7) "ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N25_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#128]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#144]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#160]\n\t"\
  "prfm pldl2keep,[x7]; add x4,x4,#400\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-224]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-208]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-192]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-176]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-160]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-144]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-128]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-112]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-96]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-80]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-64]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-48]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-32]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 3)\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(26, 27, 28, ac1, ac2, ac3, 7)

#define KERNEL_M3N25_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16\n\t"\
  FMA_M3N4(8, 9, 10, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(11, 12, 13, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(14, 15, 16, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(17, 18, 19, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(20, 21, 22, 0, 1, 2, 6, 0) "ldr s6,[x4],#4\n\t"\
  FMA_M3N4(23, 24, 25, 0, 1, 2, 7, 0) "subs w5,w5,#1\n\t"\
  "fmla v26.4s,v0.4s,v6.s[0]; fmla v27.4s,v1.4s,v6.s[0]\n\t"\
  "fmla v28.4s,v2.4s,v6.s[0]\n\t"


#define INIT_M3N26 INIT_4V(8, 9, 10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M3N26(mode) UNIT_SAVE_M3N4_##mode(8, 9, 10)\
  UNIT_SAVE_M3N4_##mode(11, 12, 13) UNIT_SAVE_M3N4_##mode(14, 15, 16)\
  UNIT_SAVE_M3N4_##mode(17, 18, 19) UNIT_SAVE_M3N4_##mode(20, 21, 22)\
  UNIT_SAVE_M3N4_##mode(23, 24, 25) EDGE_SAVE_M3N1_##mode(26, 27, 28)\
  EDGE_SAVE_M3N1_##mode(29, 30, 31)

#define KERNEL_M3N26_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16\n\t"\
  "ldr q6,[x4]; ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N26_MAIN4(ac1, ac2, ac3, an1, an2, an3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "ldr q"#an1",[x0],#16\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#128]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#144]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#160]\n\t"\
  "ldr q"#an2",[x1],#16; add x4,x4,#416\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-240]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-224]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-208]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-176]\n\t"\
  "ldr q"#an3",[x2],#16\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-144]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-96]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-80]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-32]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(26, 27, 28, ac1, ac2, ac3, 6) "ldr q6,[x4]\n\t"\
  FMA_M3N1(29, 30, 31, ac1, ac2, ac3, 7) "ldr q7,[x4,#16]\n\t"

#define KERNEL_M3N26_TAIL4(ac1, ac2, ac3) \
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#32]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#48]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#80]\n\t"\
  "prfm pldl2keep,[x6]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 0) "ldr q6,[x4,#96]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 0) "ldr q7,[x4,#112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#128]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#144]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#160]\n\t"\
  "prfm pldl2keep,[x7]; add x4,x4,#416\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-240]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 1) "ldr q6,[x4,#-224]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 1) "ldr q7,[x4,#-208]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-192]\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-176]\n\t"\
  "prfm pldl2keep,[x8]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-160]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-144]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 2) "ldr q6,[x4,#-128]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 2) "ldr q7,[x4,#-112]\n\t"\
  FMA_M3N4(8, 9, 10, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-96]\n\t"\
  "sub w5,w5,#4\n\t"\
  FMA_M3N4(11, 12, 13, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-80]\n\t"\
  FMA_M3N4(14, 15, 16, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-64]\n\t"\
  FMA_M3N4(17, 18, 19, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-48]\n\t"\
  FMA_M3N4(20, 21, 22, ac1, ac2, ac3, 6, 3) "ldr q6,[x4,#-32]\n\t"\
  FMA_M3N4(23, 24, 25, ac1, ac2, ac3, 7, 3) "ldr q7,[x4,#-16]\n\t"\
  FMA_M3N1(26, 27, 28, ac1, ac2, ac3, 6)\
  FMA_M3N1(29, 30, 31, ac1, ac2, ac3, 7)

#define KERNEL_M3N26_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4; ldr s2,[x2],#4\n\t"\
  "ldr q6,[x4],#16; ldr q7,[x4],#16\n\t"\
  FMA_M3N4(8, 9, 10, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(11, 12, 13, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(14, 15, 16, 0, 1, 2, 6, 0) "ldr q6,[x4],#16\n\t"\
  FMA_M3N4(17, 18, 19, 0, 1, 2, 7, 0) "ldr q7,[x4],#16\n\t"\
  FMA_M3N4(20, 21, 22, 0, 1, 2, 6, 0) "ldr d6,[x4],#8\n\t"\
  FMA_M3N4(23, 24, 25, 0, 1, 2, 7, 0) "subs w5,w5,#1\n\t"\
  "fmla v26.4s,v0.4s,v6.s[0]; fmla v27.4s,v1.4s,v6.s[0]\n\t"\
  "fmla v28.4s,v2.4s,v6.s[0]; fmla v29.4s,v0.4s,v6.s[1]\n\t"\
  "fmla v30.4s,v1.4s,v6.s[1]; fmla v31.4s,v2.4s,v6.s[1]\n\t"

FUNC_M3(13)
FUNC_M3(14)
FUNC_M3(15)
FUNC_M3(16)
FUNC_M3(17)
FUNC_M3(18)
FUNC_M3(19)
FUNC_M3(20)
FUNC_M3(21)
FUNC_M3(22)
FUNC_M3(23)
FUNC_M3(24)
FUNC_M3(25)
FUNC_M3(26)


#define INIT_M1N4 \
  float32x4_t cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = vdupq_n_f32(0.0f);

#define INIT_M1N5 INIT_M1N4 float32x4_t cq5 = vdupq_n_f32(0.0f);
#define INIT_M1N6 INIT_M1N5 float32x4_t cq6 = vdupq_n_f32(0.0f);
#define INIT_M1N7 INIT_M1N6 float32x4_t cq7 = vdupq_n_f32(0.0f);

#define INIT_M1N8 \
  float32x4_t cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8;\
  cq1 = cq2 = cq3 = cq4 = vdupq_n_f32(0.0f);\
  cq5 = cq6 = cq7 = cq8 = vdupq_n_f32(0.0f);

#define INIT_M1N9 INIT_M1N8 float32x4_t cq9 = vdupq_n_f32(0.0f);
#define INIT_M1N10 INIT_M1N9 float32x4_t cq10 = vdupq_n_f32(0.0f);
#define INIT_M1N11 INIT_M1N10 float32x4_t cq11 = vdupq_n_f32(0.0f);

#define INIT_M1N12 \
  float32x4_t cq1, cq2, cq3, cq4, cq5, cq6;\
  float32x4_t cq7, cq8, cq9, cq10, cq11, cq12;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = vdupq_n_f32(0.0f);\
  cq7 = cq8 = cq9 = cq10 = cq11 = cq12 = vdupq_n_f32(0.0f);

#define INIT_M1N13 INIT_M1N12 float32x4_t cq13 = vdupq_n_f32(0.0f);
#define INIT_M1N14 INIT_M1N13 float32x4_t cq14 = vdupq_n_f32(0.0f);
#define INIT_M1N15 INIT_M1N14 float32x4_t cq15 = vdupq_n_f32(0.0f);

#define INIT_M1N16 \
  float32x4_t cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = cq7 = cq8 = vdupq_n_f32(0.0f);\

#define INIT_M1N17 INIT_M1N16 float32x4_t cq9 = vdupq_n_f32(0.0f);
#define INIT_M1N18 INIT_M1N17 float32x4_t cq10 = vdupq_n_f32(0.0f);
#define INIT_M1N19 INIT_M1N18 float32x4_t cq11 = vdupq_n_f32(0.0f);

#define INIT_M1N20 \
  float32x4_t cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9, cq10;\
  cq1 = cq2 = cq3 = cq4 = cq5 = vdupq_n_f32(0.0f);\
  cq6 = cq7 = cq8 = cq9 = cq10 = vdupq_n_f32(0.0f);

#define INIT_M1N21 INIT_M1N20 float32x4_t cq11 = vdupq_n_f32(0.0f);
#define INIT_M1N22 INIT_M1N21 float32x4_t cq12 = vdupq_n_f32(0.0f);
#define INIT_M1N23 INIT_M1N22 float32x4_t cq13 = vdupq_n_f32(0.0f);

#define INIT_M1N24 \
  float32x4_t cq1, cq2, cq3, cq4, cq5, cq6;\
  float32x4_t cq7, cq8, cq9, cq10, cq11, cq12;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = vdupq_n_f32(0.0f);\
  cq7 = cq8 = cq9 = cq10 = cq11 = cq12 = vdupq_n_f32(0.0f);

#define INIT_M1N25 INIT_M1N24 float32x4_t cq13 = vdupq_n_f32(0.0f);
#define INIT_M1N26 INIT_M1N25 float32x4_t cq14 = vdupq_n_f32(0.0f);

#define ACC_K4M1N4 \
  float32x4_t aq1 = vld1q_f32(a_rd); a_rd += 4;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 0);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 1);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 2);\
  cq4 = vfmaq_laneq_f32(cq4, bq4, aq1, 3);

#define UNIT_ACC_K4M1N1(q_no, off) \
  float32x4_t bq##q_no = vld1q_f32(b_rd + off);\
  cq##q_no = vfmaq_f32(cq##q_no, bq##q_no, aq1);

#define ACC_K4M1N5 ACC_K4M1N4 UNIT_ACC_K4M1N1(5, 16)
#define ACC_K4M1N6 ACC_K4M1N5 UNIT_ACC_K4M1N1(6, 20)
#define ACC_K4M1N7 ACC_K4M1N6 UNIT_ACC_K4M1N1(7, 24)

#define ACC_K4M1N8 \
  float32x4_t aq1 = vld1q_f32(a_rd); a_rd += 4;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  float32x4_t bq5 = vld1q_f32(b_rd + 16);\
  float32x4_t bq6 = vld1q_f32(b_rd + 20);\
  float32x4_t bq7 = vld1q_f32(b_rd + 24);\
  float32x4_t bq8 = vld1q_f32(b_rd + 28);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 0);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 0);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 1);\
  cq4 = vfmaq_laneq_f32(cq4, bq4, aq1, 1);\
  cq5 = vfmaq_laneq_f32(cq5, bq5, aq1, 2);\
  cq6 = vfmaq_laneq_f32(cq6, bq6, aq1, 2);\
  cq7 = vfmaq_laneq_f32(cq7, bq7, aq1, 3);\
  cq8 = vfmaq_laneq_f32(cq8, bq8, aq1, 3);

#define ACC_K4M1N9 ACC_K4M1N8 UNIT_ACC_K4M1N1(9, 32)
#define ACC_K4M1N10 ACC_K4M1N9 UNIT_ACC_K4M1N1(10, 36)
#define ACC_K4M1N11 ACC_K4M1N10 UNIT_ACC_K4M1N1(11, 40)

#define ACC_K4M1N12 \
  float32x4_t aq1 = vld1q_f32(a_rd); a_rd += 4;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  float32x4_t bq5 = vld1q_f32(b_rd + 16);\
  float32x4_t bq6 = vld1q_f32(b_rd + 20);\
  float32x4_t bq7 = vld1q_f32(b_rd + 24);\
  float32x4_t bq8 = vld1q_f32(b_rd + 28);\
  float32x4_t bq9 = vld1q_f32(b_rd + 32);\
  float32x4_t bq10 = vld1q_f32(b_rd + 36);\
  float32x4_t bq11 = vld1q_f32(b_rd + 40);\
  float32x4_t bq12 = vld1q_f32(b_rd + 44);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 0);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 0);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 0);\
  cq4 = vfmaq_laneq_f32(cq4, bq4, aq1, 1);\
  cq5 = vfmaq_laneq_f32(cq5, bq5, aq1, 1);\
  cq6 = vfmaq_laneq_f32(cq6, bq6, aq1, 1);\
  cq7 = vfmaq_laneq_f32(cq7, bq7, aq1, 2);\
  cq8 = vfmaq_laneq_f32(cq8, bq8, aq1, 2);\
  cq9 = vfmaq_laneq_f32(cq9, bq9, aq1, 2);\
  cq10 = vfmaq_laneq_f32(cq10, bq10, aq1, 3);\
  cq11 = vfmaq_laneq_f32(cq11, bq11, aq1, 3);\
  cq12 = vfmaq_laneq_f32(cq12, bq12, aq1, 3);

#define ACC_K4M1N13 ACC_K4M1N12 UNIT_ACC_K4M1N1(13, 48)
#define ACC_K4M1N14 ACC_K4M1N13 UNIT_ACC_K4M1N1(14, 52)
#define ACC_K4M1N15 ACC_K4M1N14 UNIT_ACC_K4M1N1(15, 56)

#define ACC_K4M1N16 \
  float32x4_t aq1 = vld1q_f32(a_rd); a_rd += 4;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  float32x4_t bq5 = vld1q_f32(b_rd + 16);\
  float32x4_t bq6 = vld1q_f32(b_rd + 20);\
  float32x4_t bq7 = vld1q_f32(b_rd + 24);\
  float32x4_t bq8 = vld1q_f32(b_rd + 28);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 0); bq1 = vld1q_f32(b_rd + 32);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 0); bq2 = vld1q_f32(b_rd + 36);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 0); bq3 = vld1q_f32(b_rd + 40);\
  cq4 = vfmaq_laneq_f32(cq4, bq4, aq1, 0); bq4 = vld1q_f32(b_rd + 44);\
  cq5 = vfmaq_laneq_f32(cq5, bq5, aq1, 1); bq5 = vld1q_f32(b_rd + 48);\
  cq6 = vfmaq_laneq_f32(cq6, bq6, aq1, 1); bq6 = vld1q_f32(b_rd + 52);\
  cq7 = vfmaq_laneq_f32(cq7, bq7, aq1, 1); bq7 = vld1q_f32(b_rd + 56);\
  cq8 = vfmaq_laneq_f32(cq8, bq8, aq1, 1); bq8 = vld1q_f32(b_rd + 60);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 2);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 2);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 2);\
  cq4 = vfmaq_laneq_f32(cq4, bq4, aq1, 2);\
  cq5 = vfmaq_laneq_f32(cq5, bq5, aq1, 3);\
  cq6 = vfmaq_laneq_f32(cq6, bq6, aq1, 3);\
  cq7 = vfmaq_laneq_f32(cq7, bq7, aq1, 3);\
  cq8 = vfmaq_laneq_f32(cq8, bq8, aq1, 3);

#define ACC_K4M1N17 ACC_K4M1N16 UNIT_ACC_K4M1N1(9, 64)
#define ACC_K4M1N18 ACC_K4M1N17 UNIT_ACC_K4M1N1(10, 68)
#define ACC_K4M1N19 ACC_K4M1N18 UNIT_ACC_K4M1N1(11, 72)

#define ACC_K4M1N20 \
  float32x4_t aq1 = vld1q_f32(a_rd); a_rd += 4;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  float32x4_t bq5 = vld1q_f32(b_rd + 16);\
  float32x4_t bq6 = vld1q_f32(b_rd + 20);\
  float32x4_t bq7 = vld1q_f32(b_rd + 24);\
  float32x4_t bq8 = vld1q_f32(b_rd + 28);\
  float32x4_t bq9 = vld1q_f32(b_rd + 32);\
  float32x4_t bq10 = vld1q_f32(b_rd + 36);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 0); bq1 = vld1q_f32(b_rd + 40);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 0); bq2 = vld1q_f32(b_rd + 44);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 0); bq3 = vld1q_f32(b_rd + 48);\
  cq4 = vfmaq_laneq_f32(cq4, bq4, aq1, 0); bq4 = vld1q_f32(b_rd + 52);\
  cq5 = vfmaq_laneq_f32(cq5, bq5, aq1, 0); bq5 = vld1q_f32(b_rd + 56);\
  cq6 = vfmaq_laneq_f32(cq6, bq6, aq1, 1); bq6 = vld1q_f32(b_rd + 60);\
  cq7 = vfmaq_laneq_f32(cq7, bq7, aq1, 1); bq7 = vld1q_f32(b_rd + 64);\
  cq8 = vfmaq_laneq_f32(cq8, bq8, aq1, 1); bq8 = vld1q_f32(b_rd + 68);\
  cq9 = vfmaq_laneq_f32(cq9, bq9, aq1, 1); bq9 = vld1q_f32(b_rd + 72);\
  cq10 = vfmaq_laneq_f32(cq10, bq10, aq1, 1); bq10 = vld1q_f32(b_rd + 76);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 2);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 2);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 2);\
  cq4 = vfmaq_laneq_f32(cq4, bq4, aq1, 2);\
  cq5 = vfmaq_laneq_f32(cq5, bq5, aq1, 2);\
  cq6 = vfmaq_laneq_f32(cq6, bq6, aq1, 3);\
  cq7 = vfmaq_laneq_f32(cq7, bq7, aq1, 3);\
  cq8 = vfmaq_laneq_f32(cq8, bq8, aq1, 3);\
  cq9 = vfmaq_laneq_f32(cq9, bq9, aq1, 3);\
  cq10 = vfmaq_laneq_f32(cq10, bq10, aq1, 3);

#define ACC_K4M1N21 ACC_K4M1N20 UNIT_ACC_K4M1N1(11, 80)
#define ACC_K4M1N22 ACC_K4M1N21 UNIT_ACC_K4M1N1(12, 84)
#define ACC_K4M1N23 ACC_K4M1N22 UNIT_ACC_K4M1N1(13, 88)

#define ACC_K4M1N24 \
  float32x4_t aq1 = vld1q_f32(a_rd); a_rd += 4;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  float32x4_t bq5 = vld1q_f32(b_rd + 16);\
  float32x4_t bq6 = vld1q_f32(b_rd + 20);\
  float32x4_t bq7 = vld1q_f32(b_rd + 24);\
  float32x4_t bq8 = vld1q_f32(b_rd + 28);\
  float32x4_t bq9 = vld1q_f32(b_rd + 32);\
  float32x4_t bq10 = vld1q_f32(b_rd + 36);\
  float32x4_t bq11 = vld1q_f32(b_rd + 40);\
  float32x4_t bq12 = vld1q_f32(b_rd + 44);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 0); bq1 = vld1q_f32(b_rd + 48);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 0); bq2 = vld1q_f32(b_rd + 52);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 0); bq3 = vld1q_f32(b_rd + 56);\
  cq4 = vfmaq_laneq_f32(cq4, bq4, aq1, 0); bq4 = vld1q_f32(b_rd + 60);\
  cq5 = vfmaq_laneq_f32(cq5, bq5, aq1, 0); bq5 = vld1q_f32(b_rd + 64);\
  cq6 = vfmaq_laneq_f32(cq6, bq6, aq1, 0); bq6 = vld1q_f32(b_rd + 68);\
  cq7 = vfmaq_laneq_f32(cq7, bq7, aq1, 1); bq7 = vld1q_f32(b_rd + 72);\
  cq8 = vfmaq_laneq_f32(cq8, bq8, aq1, 1); bq8 = vld1q_f32(b_rd + 76);\
  cq9 = vfmaq_laneq_f32(cq9, bq9, aq1, 1); bq9 = vld1q_f32(b_rd + 80);\
  cq10 = vfmaq_laneq_f32(cq10, bq10, aq1, 1); bq10 = vld1q_f32(b_rd + 84);\
  cq11 = vfmaq_laneq_f32(cq11, bq11, aq1, 1); bq11 = vld1q_f32(b_rd + 88);\
  cq12 = vfmaq_laneq_f32(cq12, bq12, aq1, 1); bq12 = vld1q_f32(b_rd + 92);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 2);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 2);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 2);\
  cq4 = vfmaq_laneq_f32(cq4, bq4, aq1, 2);\
  cq5 = vfmaq_laneq_f32(cq5, bq5, aq1, 2);\
  cq6 = vfmaq_laneq_f32(cq6, bq6, aq1, 2);\
  cq7 = vfmaq_laneq_f32(cq7, bq7, aq1, 3);\
  cq8 = vfmaq_laneq_f32(cq8, bq8, aq1, 3);\
  cq9 = vfmaq_laneq_f32(cq9, bq9, aq1, 3);\
  cq10 = vfmaq_laneq_f32(cq10, bq10, aq1, 3);\
  cq11 = vfmaq_laneq_f32(cq11, bq11, aq1, 3);\
  cq12 = vfmaq_laneq_f32(cq12, bq12, aq1, 3);

#define ACC_K4M1N25 ACC_K4M1N24 UNIT_ACC_K4M1N1(13, 96)
#define ACC_K4M1N26 ACC_K4M1N25 UNIT_ACC_K4M1N1(14, 100)

#define REDUC_M1N4 \
  cq1 = vaddq_f32(cq1, cq2); cq3 = vaddq_f32(cq3, cq4);\
  cq1 = vaddq_f32(cq1, cq3);

#define UNIT_REDUC_1V(q_no, s_no) \
  float32x2_t cd##s_no = vadd_f32(vget_low_f32(cq##q_no),\
    vget_high_f32(cq##q_no));\
  float cs##s_no = vget_lane_f32(cd##s_no, 0) + vget_lane_f32(cd##s_no, 1);

#define REDUC_M1N5 REDUC_M1N4 UNIT_REDUC_1V(5, 1)
#define REDUC_M1N6 REDUC_M1N5 UNIT_REDUC_1V(6, 2)
#define REDUC_M1N7 REDUC_M1N6 UNIT_REDUC_1V(7, 3)

#define REDUC_M1N8 \
  cq1 = vaddq_f32(cq1, cq3); cq2 = vaddq_f32(cq2, cq4);\
  cq5 = vaddq_f32(cq5, cq7); cq6 = vaddq_f32(cq6, cq8);\
  cq1 = vaddq_f32(cq1, cq5); cq2 = vaddq_f32(cq2, cq6);

#define REDUC_M1N9 REDUC_M1N8 UNIT_REDUC_1V(9, 1)
#define REDUC_M1N10 REDUC_M1N9 UNIT_REDUC_1V(10, 2)
#define REDUC_M1N11 REDUC_M1N10 UNIT_REDUC_1V(11, 3)

#define REDUC_M1N12 \
  cq1 = vaddq_f32(cq1, cq4); cq2 = vaddq_f32(cq2, cq5);\
  cq3 = vaddq_f32(cq3, cq6); cq7 = vaddq_f32(cq7, cq10);\
  cq8 = vaddq_f32(cq8, cq11); cq9 = vaddq_f32(cq9, cq12);\
  cq1 = vaddq_f32(cq1, cq7); cq2 = vaddq_f32(cq2, cq8);\
  cq3 = vaddq_f32(cq3, cq9);

#define REDUC_M1N13 REDUC_M1N12 UNIT_REDUC_1V(13, 1)
#define REDUC_M1N14 REDUC_M1N13 UNIT_REDUC_1V(14, 2)
#define REDUC_M1N15 REDUC_M1N14 UNIT_REDUC_1V(15, 3)

#define REDUC_M1N16 \
  cq1 = vaddq_f32(cq1, cq5); cq2 = vaddq_f32(cq2, cq6);\
  cq3 = vaddq_f32(cq3, cq7); cq4 = vaddq_f32(cq4, cq8);

#define REDUC_M1N17 REDUC_M1N16 UNIT_REDUC_1V(9, 1)
#define REDUC_M1N18 REDUC_M1N17 UNIT_REDUC_1V(10, 2)
#define REDUC_M1N19 REDUC_M1N18 UNIT_REDUC_1V(11, 3)

#define REDUC_M1N20 \
  cq1 = vaddq_f32(cq1, cq6); cq2 = vaddq_f32(cq2, cq7);\
  cq3 = vaddq_f32(cq3, cq8); cq4 = vaddq_f32(cq4, cq9);\
  cq5 = vaddq_f32(cq5, cq10);

#define REDUC_M1N21 REDUC_M1N20 UNIT_REDUC_1V(11, 1)
#define REDUC_M1N22 REDUC_M1N21 UNIT_REDUC_1V(12, 2)
#define REDUC_M1N23 REDUC_M1N22 UNIT_REDUC_1V(13, 3)

#define REDUC_M1N24 \
  cq1 = vaddq_f32(cq1, cq7); cq2 = vaddq_f32(cq2, cq8);\
  cq3 = vaddq_f32(cq3, cq9); cq4 = vaddq_f32(cq4, cq10);\
  cq5 = vaddq_f32(cq5, cq11); cq6 = vaddq_f32(cq6, cq12);

#define REDUC_M1N25 REDUC_M1N24 UNIT_REDUC_1V(13, 1)
#define REDUC_M1N26 REDUC_M1N25 UNIT_REDUC_1V(14, 2)

#define ACC_K1M1N4 \
  float as1 = *a_rd++;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  cq1 = vfmaq_n_f32(cq1, bq1, as1);

#define ACC_K1M1N5 ACC_K1M1N4 cs1 += as1 * b_rd[4];
#define ACC_K1M1N6 ACC_K1M1N5 cs2 += as1 * b_rd[5];
#define ACC_K1M1N7 ACC_K1M1N6 cs3 += as1 * b_rd[6];

#define ACC_K1M1N8 \
  float as1 = *a_rd++;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  cq1 = vfmaq_n_f32(cq1, bq1, as1);\
  cq2 = vfmaq_n_f32(cq2, bq2, as1);

#define ACC_K1M1N9 ACC_K1M1N8 cs1 += as1 * b_rd[8];
#define ACC_K1M1N10 ACC_K1M1N9 cs2 += as1 * b_rd[9];
#define ACC_K1M1N11 ACC_K1M1N10 cs3 += as1 * b_rd[10];

#define ACC_K1M1N12 \
  float as1 = *a_rd++;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  cq1 = vfmaq_n_f32(cq1, bq1, as1);\
  cq2 = vfmaq_n_f32(cq2, bq2, as1);\
  cq3 = vfmaq_n_f32(cq3, bq3, as1);

#define ACC_K1M1N13 ACC_K1M1N12 cs1 += as1 * b_rd[12];
#define ACC_K1M1N14 ACC_K1M1N13 cs2 += as1 * b_rd[13];
#define ACC_K1M1N15 ACC_K1M1N14 cs3 += as1 * b_rd[14];

#define ACC_K1M1N16 \
  float as1 = *a_rd++;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  cq1 = vfmaq_n_f32(cq1, bq1, as1);\
  cq2 = vfmaq_n_f32(cq2, bq2, as1);\
  cq3 = vfmaq_n_f32(cq3, bq3, as1);\
  cq4 = vfmaq_n_f32(cq4, bq4, as1);

#define ACC_K1M1N17 ACC_K1M1N16 cs1 += as1 * b_rd[16];
#define ACC_K1M1N18 ACC_K1M1N17 cs2 += as1 * b_rd[17];
#define ACC_K1M1N19 ACC_K1M1N18 cs3 += as1 * b_rd[18];

#define ACC_K1M1N20 \
  float as1 = *a_rd++;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  float32x4_t bq5 = vld1q_f32(b_rd + 16);\
  cq1 = vfmaq_n_f32(cq1, bq1, as1);\
  cq2 = vfmaq_n_f32(cq2, bq2, as1);\
  cq3 = vfmaq_n_f32(cq3, bq3, as1);\
  cq4 = vfmaq_n_f32(cq4, bq4, as1);\
  cq5 = vfmaq_n_f32(cq5, bq5, as1);

#define ACC_K1M1N21 ACC_K1M1N20 cs1 += as1 * b_rd[20];
#define ACC_K1M1N22 ACC_K1M1N21 cs2 += as1 * b_rd[21];
#define ACC_K1M1N23 ACC_K1M1N22 cs3 += as1 * b_rd[22];

#define ACC_K1M1N24 \
  float as1 = *a_rd++;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  float32x4_t bq5 = vld1q_f32(b_rd + 16);\
  float32x4_t bq6 = vld1q_f32(b_rd + 20);\
  cq1 = vfmaq_n_f32(cq1, bq1, as1);\
  cq2 = vfmaq_n_f32(cq2, bq2, as1);\
  cq3 = vfmaq_n_f32(cq3, bq3, as1);\
  cq4 = vfmaq_n_f32(cq4, bq4, as1);\
  cq5 = vfmaq_n_f32(cq5, bq5, as1);\
  cq6 = vfmaq_n_f32(cq6, bq6, as1);

#define ACC_K1M1N25 ACC_K1M1N24 cs1 += as1 * b_rd[24];
#define ACC_K1M1N26 ACC_K1M1N25 cs2 += as1 * b_rd[25];

#define UNIT_SAVE_M1N4_CC(cq1) \
  c_ptr[0] = c_ptr[0] * beta + vgetq_lane_f32(cq1, 0);\
  c_ptr[LDC] = c_ptr[LDC] * beta + vgetq_lane_f32(cq1, 1);\
  c_ptr += LDC * 2;\
  c_ptr[0] = c_ptr[0] * beta + vgetq_lane_f32(cq1, 2);\
  c_ptr[LDC] = c_ptr[LDC] * beta + vgetq_lane_f32(cq1, 3);\
  c_ptr += LDC * 2;

#define UNIT_SAVE_M1N4_CR(cq1) \
  cq1 = vfmaq_n_f32(cq1, vld1q_f32(c_ptr), beta);\
  vst1q_f32(c_ptr, cq1); c_ptr += 4;

#define UNIT_SAVE_M1N1_CC(cs1) \
  c_ptr[0] = c_ptr[0] * beta + cs1; c_ptr += LDC;

#define UNIT_SAVE_M1N1_CR(cs1) \
  c_ptr[0] = c_ptr[0] * beta + cs1; c_ptr++;

#define SAVE_M1N4(mode) UNIT_SAVE_M1N4_##mode(cq1)

#define SAVE_M1N5(mode) SAVE_M1N4(mode) UNIT_SAVE_M1N1_##mode(cs1)
#define SAVE_M1N6(mode) SAVE_M1N5(mode) UNIT_SAVE_M1N1_##mode(cs2)
#define SAVE_M1N7(mode) SAVE_M1N6(mode) UNIT_SAVE_M1N1_##mode(cs3)

#define SAVE_M1N8(mode) \
  UNIT_SAVE_M1N4_##mode(cq1) UNIT_SAVE_M1N4_##mode(cq2)

#define SAVE_M1N9(mode) SAVE_M1N8(mode) UNIT_SAVE_M1N1_##mode(cs1)
#define SAVE_M1N10(mode) SAVE_M1N9(mode) UNIT_SAVE_M1N1_##mode(cs2)
#define SAVE_M1N11(mode) SAVE_M1N10(mode) UNIT_SAVE_M1N1_##mode(cs3)

#define SAVE_M1N12(mode) \
  UNIT_SAVE_M1N4_##mode(cq1) UNIT_SAVE_M1N4_##mode(cq2) UNIT_SAVE_M1N4_##mode(cq3)

#define SAVE_M1N13(mode) SAVE_M1N12(mode) UNIT_SAVE_M1N1_##mode(cs1)
#define SAVE_M1N14(mode) SAVE_M1N13(mode) UNIT_SAVE_M1N1_##mode(cs2)
#define SAVE_M1N15(mode) SAVE_M1N14(mode) UNIT_SAVE_M1N1_##mode(cs3)

#define SAVE_M1N16(mode) \
  UNIT_SAVE_M1N4_##mode(cq1) UNIT_SAVE_M1N4_##mode(cq2)\
  UNIT_SAVE_M1N4_##mode(cq3) UNIT_SAVE_M1N4_##mode(cq4)

#define SAVE_M1N17(mode) SAVE_M1N16(mode) UNIT_SAVE_M1N1_##mode(cs1)
#define SAVE_M1N18(mode) SAVE_M1N17(mode) UNIT_SAVE_M1N1_##mode(cs2)
#define SAVE_M1N19(mode) SAVE_M1N18(mode) UNIT_SAVE_M1N1_##mode(cs3)

#define SAVE_M1N20(mode) \
  UNIT_SAVE_M1N4_##mode(cq1) UNIT_SAVE_M1N4_##mode(cq2)\
  UNIT_SAVE_M1N4_##mode(cq3) UNIT_SAVE_M1N4_##mode(cq4) UNIT_SAVE_M1N4_##mode(cq5)

#define SAVE_M1N21(mode) SAVE_M1N20(mode) UNIT_SAVE_M1N1_##mode(cs1)
#define SAVE_M1N22(mode) SAVE_M1N21(mode) UNIT_SAVE_M1N1_##mode(cs2)
#define SAVE_M1N23(mode) SAVE_M1N22(mode) UNIT_SAVE_M1N1_##mode(cs3)

#define SAVE_M1N24(mode) \
  UNIT_SAVE_M1N4_##mode(cq1) UNIT_SAVE_M1N4_##mode(cq2) UNIT_SAVE_M1N4_##mode(cq3)\
  UNIT_SAVE_M1N4_##mode(cq4) UNIT_SAVE_M1N4_##mode(cq5) UNIT_SAVE_M1N4_##mode(cq6)

#define SAVE_M1N25(mode) SAVE_M1N24(mode) UNIT_SAVE_M1N1_##mode(cs1)
#define SAVE_M1N26(mode) SAVE_M1N25(mode) UNIT_SAVE_M1N1_##mode(cs2)

#define FUNC_M1(ndim) \
static inline void sgemm_skinny1_a7x_m1n##ndim(\
  const float * __restrict__ a_rd, const float * __restrict__ b_rd,\
  float * __restrict__ c_ptr, uint32_t k_left, uint32_t LDC,\
  uint8_t c_rowmajor, float beta) {\
  INIT_M1N##ndim\
  for (; k_left > 3; k_left -= 4) {\
    ACC_K4M1N##ndim\
    b_rd += 4 * ndim;\
  }\
  REDUC_M1N##ndim\
  for (; k_left > 0; k_left--) {\
    ACC_K1M1N##ndim\
    b_rd += ndim;\
  }\
  if (c_rowmajor == 0) {\
    SAVE_M1N##ndim(CC)\
  } else {\
    SAVE_M1N##ndim(CR)\
  }\
}

FUNC_M1(4)
FUNC_M1(5)
FUNC_M1(6)
FUNC_M1(7)
FUNC_M1(8)
FUNC_M1(9)
FUNC_M1(10)
FUNC_M1(11)
FUNC_M1(12)
FUNC_M1(13)
FUNC_M1(14)
FUNC_M1(15)
FUNC_M1(16)
FUNC_M1(17)
FUNC_M1(18)
FUNC_M1(19)
FUNC_M1(20)
FUNC_M1(21)
FUNC_M1(22)
FUNC_M1(23)
FUNC_M1(24)
FUNC_M1(25)
FUNC_M1(26)

#endif
