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


#include "common/CommonKernel.h"
#include "arm_neon/NeonSgemmKernel.h"

#define NEON_SGEMM_KERNEL_M6N8_PRELOAD_A53 \
  "vldr d0,[%13]; vldr d1,[%13,#8]; add %13,%13,#24\n\t"\
  "vldr d4,[%14]; vldr d5,[%14,#8]; ldr r2,[%14,#16]; ldr r3,[%14,#20]\n\t"\
  "add %14,%14,#32\n\t"

#define NEON_SGEMM_KERNEL_M6N8_MAIN2_A53 \
  "vldr d3,[%13,#-8]; vmov d2,d1\n\t"\
  "vmla.f32 %q0,q0,d4[0]; ldr r0,[%14]\n\t"\
  "vmla.f32 %q1,q0,d4[1]; ldr r1,[%14,#4]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d7,[%14,#-8]; vmov d6,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]; ldr r2,[%13]\n\t"\
  "vmla.f32 %q4,q2,d3[0]; ldr r3,[%13,#4]\n\t"\
  "vmla.f32 %q5,q2,d3[1]\n\t"\
  "vldr d5,[%14,#8]; vmov d4,r0,r1\n\t"\
  "vmla.f32 %q6,q3,d0[0]; add %13,%13,#48\n\t"\
  "vmla.f32 %q7,q3,d0[1]; add %14,%14,#64\n\t"\
  "vmla.f32 %q8,q1,d6[0]; pld [%13,#128]\n\t"\
  "vldr d1,[%13,#-40]; vmov d0,r2,r3\n\t"\
  "vmla.f32 %q9,q1,d6[1]; ldr r2,[%14,#-48]\n\t"\
  "vmla.f32 %q10,q1,d7[0]; ldr r3,[%14,#-44]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"\
  "vldr d3,[%13,#-32]; vmov d2,d1\n\t"\
  "vmla.f32 %q0,q0,d4[0]; ldr r0,[%14,#-32]\n\t"\
  "vmla.f32 %q1,q0,d4[1]; ldr r1,[%14,#-28]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d7,[%14,#-40]; vmov d6,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]; ldr r2,[%13,#-24]\n\t"\
  "vmla.f32 %q4,q2,d3[0]; ldr r3,[%13,#-20]\n\t"\
  "vmla.f32 %q5,q2,d3[1]\n\t"\
  "vldr d5,[%14,#-24]; vmov d4,r0,r1\n\t"\
  "vmla.f32 %q6,q3,d0[0]; sub %12,%12,#2\n\t"\
  "vmla.f32 %q7,q3,d0[1]; cmp %12,#2\n\t"\
  "vmla.f32 %q8,q1,d6[0]; pld [%14,#192]\n\t"\
  "vldr d1,[%13,#-16]; vmov d0,r2,r3\n\t"\
  "vmla.f32 %q9,q1,d6[1]; ldr r2,[%14,#-16]\n\t"\
  "vmla.f32 %q10,q1,d7[0]; ldr r3,[%14,#-12]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"

#define NEON_SGEMM_KERNEL_M6N8_TAIL2_A53 \
  "vldr d3,[%13,#-8]; vmov d2,d1\n\t"\
  "vmla.f32 %q0,q0,d4[0]; ldr r0,[%14]\n\t"\
  "vmla.f32 %q1,q0,d4[1]; ldr r1,[%14,#4]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d7,[%14,#-8]; vmov d6,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]; ldr r2,[%13]\n\t"\
  "vmla.f32 %q4,q2,d3[0]; ldr r3,[%13,#4]\n\t"\
  "vmla.f32 %q5,q2,d3[1]\n\t"\
  "vldr d5,[%14,#8]; vmov d4,r0,r1\n\t"\
  "vmla.f32 %q6,q3,d0[0]; add %13,%13,#24\n\t"\
  "vmla.f32 %q7,q3,d0[1]; add %14,%14,#32\n\t"\
  "vmla.f32 %q8,q1,d6[0]\n\t"\
  "vldr d1,[%13,#-16]; vmov d0,r2,r3\n\t"\
  "vmla.f32 %q9,q1,d6[1]; ldr r2,[%14,#-16]\n\t"\
  "vmla.f32 %q10,q1,d7[0]; ldr r3,[%14,#-12]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"\
  "vldr d3,[%13,#-8]; vmov d2,d1\n\t"\
  "vmla.f32 %q0,q0,d4[0]\n\t"\
  "vmla.f32 %q1,q0,d4[1]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d7,[%14,#-8]; vmov d6,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]\n\t"\
  "vmla.f32 %q4,q2,d3[0]\n\t"\
  "vmla.f32 %q5,q2,d3[1]\n\t"\
  "vmla.f32 %q6,q3,d0[0]\n\t"\
  "vmla.f32 %q7,q3,d0[1]\n\t"\
  "vmla.f32 %q8,q1,d6[0]\n\t"\
  "vmla.f32 %q9,q1,d6[1]\n\t"\
  "vmla.f32 %q10,q1,d7[0]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"

#define NEON_SGEMM_KERNEL_M6N8_TAIL1_A53 \
  "vldr d3,[%13,#-8]; vmov d2,d1\n\t"\
  "vmla.f32 %q0,q0,d4[0]\n\t"\
  "vmla.f32 %q1,q0,d4[1]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d7,[%14,#-8]; vmov d6,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]\n\t"\
  "vmla.f32 %q4,q2,d3[0]\n\t"\
  "vmla.f32 %q5,q2,d3[1]\n\t"\
  "vmla.f32 %q6,q3,d0[0]\n\t"\
  "vmla.f32 %q7,q3,d0[1]\n\t"\
  "vmla.f32 %q8,q1,d6[0]\n\t"\
  "vmla.f32 %q9,q1,d6[1]\n\t"\
  "vmla.f32 %q10,q1,d7[0]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"

#define NEON_SGEMM_SAVE_M6N8_ASM \
  float32x4x2_t ct1 = vzipq_f32(cq05, cq06);\
  float32x2_t cd1 = vget_low_f32(ct1.val[0]);\
  float32x2_t cd2 = vget_high_f32(ct1.val[0]);\
\
  cq01 = vmlaq_n_f32(cq01, vld1q_f32(c_tmp), beta);\
  cd1 = vmla_n_f32(cd1, vld1_f32(c_tmp + 4), beta);\
  cq02 = vmlaq_n_f32(cq02, vld1q_f32(c_tmp + ldc), beta);\
  cd2 = vmla_n_f32(cd2, vld1_f32(c_tmp + ldc + 4), beta);\
\
  vst1q_f32(c_tmp, cq01); vst1_f32(c_tmp + 4, cd1); c_tmp += ldc;\
  vst1q_f32(c_tmp, cq02); vst1_f32(c_tmp + 4, cd2); c_tmp += ldc;\
  cd1 = vget_low_f32(ct1.val[1]);\
  cd2 = vget_high_f32(ct1.val[1]);\
\
  cq03 = vmlaq_n_f32(cq03, vld1q_f32(c_tmp), beta);\
  cd1 = vmla_n_f32(cd1, vld1_f32(c_tmp + 4), beta);\
  cq04 = vmlaq_n_f32(cq04, vld1q_f32(c_tmp + ldc), beta);\
  cd2 = vmla_n_f32(cd2, vld1_f32(c_tmp + ldc + 4), beta);\
\
  vst1q_f32(c_tmp, cq03); vst1_f32(c_tmp + 4, cd1); c_tmp += ldc;\
  vst1q_f32(c_tmp, cq04); vst1_f32(c_tmp + 4, cd2); c_tmp += ldc;\
  ct1 = vzipq_f32(cq07, cq08);\
  cd1 = vget_low_f32(ct1.val[0]);\
  cd2 = vget_high_f32(ct1.val[0]);\
\
  cd1 = vmla_n_f32(cd1, vld1_f32(c_tmp), beta);\
  cq09 = vmlaq_n_f32(cq09, vld1q_f32(c_tmp + 2), beta);\
  cd2 = vmla_n_f32(cd2, vld1_f32(c_tmp + ldc), beta);\
  cq10 = vmlaq_n_f32(cq10, vld1q_f32(c_tmp + ldc + 2), beta);\
\
  vst1_f32(c_tmp, cd1); vst1q_f32(c_tmp + 2, cq09); c_tmp += ldc;\
  vst1_f32(c_tmp, cd2); vst1q_f32(c_tmp + 2, cq10); c_tmp += ldc;\
  cd1 = vget_low_f32(ct1.val[1]);\
  cd2 = vget_high_f32(ct1.val[1]);\
\
  cd1 = vmla_n_f32(cd1, vld1_f32(c_tmp), beta);\
  cq11 = vmlaq_n_f32(cq11, vld1q_f32(c_tmp + 2), beta);\
  cd2 = vmla_n_f32(cd2, vld1_f32(c_tmp + ldc), beta);\
  cq12 = vmlaq_n_f32(cq12, vld1q_f32(c_tmp + ldc + 2), beta);\
\
  vst1_f32(c_tmp, cd1); vst1q_f32(c_tmp + 2, cq11); c_tmp += ldc;\
  vst1_f32(c_tmp, cd2); vst1q_f32(c_tmp + 2, cq12);

#define NEON_SGEMM_KERNEL_M8N6_PRELOAD_A53 \
  "vldr d0,[%13]; vldr d1,[%13,#8]\n\t"\
  "ldr r2,[%13,#16]; ldr r3,[%13,#20]; add %13,%13,#32\n\t"\
  "vldr d4,[%14]; vldr d5,[%14,#8]; add %14,%14,#24\n\t"

#define NEON_SGEMM_KERNEL_M8N6_MAIN2_A53 \
  "vldr d7,[%14,#-8]; vmov d6,d5\n\t"\
  "vmla.f32 %q0,q0,d4[0]; ldr r0,[%13]\n\t"\
  "vmla.f32 %q1,q0,d4[1]; ldr r1,[%13,#4]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d3,[%13,#-8]; vmov d2,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]; ldr r2,[%14]\n\t"\
  "vmla.f32 %q4,q0,d7[0]; ldr r3,[%14,#4]\n\t"\
  "vmla.f32 %q5,q0,d7[1]\n\t"\
  "vldr d1,[%13,#8]; vmov d0,r0,r1\n\t"\
  "vmla.f32 %q6,q1,d4[0]; add %13,%13,#64\n\t"\
  "vmla.f32 %q7,q1,d4[1]; add %14,%14,#48\n\t"\
  "vmla.f32 %q8,q1,d6[0]; pld [%13,#192]\n\t"\
  "vldr d5,[%14,#-40]; vmov d4,r2,r3\n\t"\
  "vmla.f32 %q9,q1,d6[1]; ldr r2,[%13,#-48]\n\t"\
  "vmla.f32 %q10,q1,d7[0]; ldr r3,[%13,#-44]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"\
  "vldr d7,[%14,#-32]; vmov d6,d5\n\t"\
  "vmla.f32 %q0,q0,d4[0]; ldr r0,[%13,#-32]\n\t"\
  "vmla.f32 %q1,q0,d4[1]; ldr r1,[%13,#-28]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d3,[%13,#-40]; vmov d2,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]; ldr r2,[%14,#-24]\n\t"\
  "vmla.f32 %q4,q0,d7[0]; ldr r3,[%14,#-20]\n\t"\
  "vmla.f32 %q5,q0,d7[1]\n\t"\
  "vldr d1,[%13,#-24]; vmov d0,r0,r1\n\t"\
  "vmla.f32 %q6,q1,d4[0]; sub %12,%12,#2\n\t"\
  "vmla.f32 %q7,q1,d4[1]; cmp %12,#2\n\t"\
  "vmla.f32 %q8,q1,d6[0]; pld [%14,#128]\n\t"\
  "vldr d5,[%14,#-16]; vmov d4,r2,r3\n\t"\
  "vmla.f32 %q9,q1,d6[1]; ldr r2,[%13,#-16]\n\t"\
  "vmla.f32 %q10,q1,d7[0]; ldr r3,[%13,#-12]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"

#define NEON_SGEMM_KERNEL_M8N6_TAIL2_A53 \
  "vldr d7,[%14,#-8]; vmov d6,d5\n\t"\
  "vmla.f32 %q0,q0,d4[0]; ldr r0,[%13]\n\t"\
  "vmla.f32 %q1,q0,d4[1]; ldr r1,[%13,#4]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d3,[%13,#-8]; vmov d2,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]; ldr r2,[%14]\n\t"\
  "vmla.f32 %q4,q0,d7[0]; ldr r3,[%14,#4]\n\t"\
  "vmla.f32 %q5,q0,d7[1]\n\t"\
  "vldr d1,[%13,#8]; vmov d0,r0,r1\n\t"\
  "vmla.f32 %q6,q1,d4[0]; add %13,%13,#32\n\t"\
  "vmla.f32 %q7,q1,d4[1]; add %14,%14,#24\n\t"\
  "vmla.f32 %q8,q1,d6[0]\n\t"\
  "vldr d5,[%14,#-16]; vmov d4,r2,r3\n\t"\
  "vmla.f32 %q9,q1,d6[1]; ldr r2,[%13,#-16]\n\t"\
  "vmla.f32 %q10,q1,d7[0]; ldr r3,[%13,#-12]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"\
  "vldr d7,[%14,#-8]; vmov d6,d5\n\t"\
  "vmla.f32 %q0,q0,d4[0]\n\t"\
  "vmla.f32 %q1,q0,d4[1]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d3,[%13,#-8]; vmov d2,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]\n\t"\
  "vmla.f32 %q4,q0,d7[0]\n\t"\
  "vmla.f32 %q5,q0,d7[1]\n\t"\
  "vmla.f32 %q6,q1,d4[0]\n\t"\
  "vmla.f32 %q7,q1,d4[1]\n\t"\
  "vmla.f32 %q8,q1,d6[0]\n\t"\
  "vmla.f32 %q9,q1,d6[1]\n\t"\
  "vmla.f32 %q10,q1,d7[0]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"

#define NEON_SGEMM_KERNEL_M8N6_TAIL1_A53 \
  "vldr d7,[%14,#-8]; vmov d6,d5\n\t"\
  "vmla.f32 %q0,q0,d4[0]\n\t"\
  "vmla.f32 %q1,q0,d4[1]\n\t"\
  "vmla.f32 %q2,q0,d5[0]\n\t"\
  "vldr d3,[%13,#-8]; vmov d2,r2,r3\n\t"\
  "vmla.f32 %q3,q0,d5[1]\n\t"\
  "vmla.f32 %q4,q0,d7[0]\n\t"\
  "vmla.f32 %q5,q0,d7[1]\n\t"\
  "vmla.f32 %q6,q1,d4[0]\n\t"\
  "vmla.f32 %q7,q1,d4[1]\n\t"\
  "vmla.f32 %q8,q1,d6[0]\n\t"\
  "vmla.f32 %q9,q1,d6[1]\n\t"\
  "vmla.f32 %q10,q1,d7[0]\n\t"\
  "vmla.f32 %q11,q1,d7[1]\n\t"

#define NEON_SGEMM_SAVE_M8N6_ASM \
\
  cq01 = vmlaq_n_f32(cq01, vld1q_f32(c_tmp), beta);\
  cq07 = vmlaq_n_f32(cq07, vld1q_f32(c_tmp + 4), beta);\
  cq02 = vmlaq_n_f32(cq02, vld1q_f32(c_tmp + ldc), beta);\
  cq08 = vmlaq_n_f32(cq08, vld1q_f32(c_tmp + ldc + 4), beta);\
\
  vst1q_f32(c_tmp, cq01); vst1q_f32(c_tmp + 4, cq07); c_tmp += ldc;\
  vst1q_f32(c_tmp, cq02); vst1q_f32(c_tmp + 4, cq08); c_tmp += ldc;\
\
  cq03 = vmlaq_n_f32(cq03, vld1q_f32(c_tmp), beta);\
  cq09 = vmlaq_n_f32(cq09, vld1q_f32(c_tmp + 4), beta);\
  cq04 = vmlaq_n_f32(cq04, vld1q_f32(c_tmp + ldc), beta);\
  cq10 = vmlaq_n_f32(cq10, vld1q_f32(c_tmp + ldc + 4), beta);\
\
  vst1q_f32(c_tmp, cq03); vst1q_f32(c_tmp + 4, cq09); c_tmp += ldc;\
  vst1q_f32(c_tmp, cq04); vst1q_f32(c_tmp + 4, cq10); c_tmp += ldc;\
\
  cq05 = vmlaq_n_f32(cq05, vld1q_f32(c_tmp), beta);\
  cq11 = vmlaq_n_f32(cq11, vld1q_f32(c_tmp + 4), beta);\
  cq06 = vmlaq_n_f32(cq06, vld1q_f32(c_tmp + ldc), beta);\
  cq12 = vmlaq_n_f32(cq12, vld1q_f32(c_tmp + ldc + 4), beta);\
\
  vst1q_f32(c_tmp, cq05); vst1q_f32(c_tmp + 4, cq11); c_tmp += ldc;\
  vst1q_f32(c_tmp, cq06); vst1q_f32(c_tmp + 4, cq12);

#define PREF_C_1_LANE(n, mdim) \
  pref_c(c_pref); pref_c(c_pref + mdim - 1); c_pref += ldc;
#define PREF_C(mdim, ndim) \
  MACRO_EXPANSION_##ndim(VOID_BASE, PREF_C_1_LANE, mdim)

#define NEON_SGEMM_ASM(mdim, ndim, cputype) {\
  float *c_pref = c_ptr; PREF_C(mdim, ndim)\
  register float32x4_t cq01 __asm("q4");\
  register float32x4_t cq02 __asm("q5");\
  register float32x4_t cq03 __asm("q6");\
  register float32x4_t cq04 __asm("q7");\
  register float32x4_t cq05 __asm("q8");\
  register float32x4_t cq06 __asm("q9");\
  register float32x4_t cq07 __asm("q10");\
  register float32x4_t cq08 __asm("q11");\
  register float32x4_t cq09 __asm("q12");\
  register float32x4_t cq10 __asm("q13");\
  register float32x4_t cq11 __asm("q14");\
  register float32x4_t cq12 __asm("q15");\
  const float *a_ptr, *b_ptr;\
  uint32_t k_left;\
  b_ptr = b_head;\
  a_ptr = a_head;\
  k_left = K;\
  __asm__ __volatile__ (\
    "vmov.i8 %q0,#0; vmov.i8 %q1,#0; vmov %q2,%q0; vmov %q3,%q1\n\t"\
    "vmov %q4,%q0; vmov %q5,%q1; vmov %q6,%q0; vmov %q7,%q1\n\t"\
    "vmov %q8,%q0; vmov %q9,%q1; vmov %q10,%q0; vmov %q11,%q1\n\t"\
    "cmp %12,#0; beq 4f\n\t"\
    NEON_SGEMM_KERNEL_M##mdim##N##ndim##_PRELOAD_##cputype\
    "cmp %12,#2; ble 2f\n\t"\
    ".balign 16\n\t"\
    "1:\n\t"\
    NEON_SGEMM_KERNEL_M##mdim##N##ndim##_MAIN2_##cputype "bgt 1b\n\t"\
    "2:\n\t"\
    "cmp %12,#2; bne 3f\n\t"\
    NEON_SGEMM_KERNEL_M##mdim##N##ndim##_TAIL2_##cputype "b 4f\n\t"\
    "3:\n\t"\
    NEON_SGEMM_KERNEL_M##mdim##N##ndim##_TAIL1_##cputype\
    "4:\n\t"\
  :"=w"(cq01),"=w"(cq02),"=w"(cq03),"=w"(cq04),"=w"(cq05),"=w"(cq06),\
  "=w"(cq07),"=w"(cq08),"=w"(cq09),"=w"(cq10),"=w"(cq11),"=w"(cq12),\
  "+r"(k_left),"+r"(a_ptr),"+r"(b_ptr)\
  ::"d0","d1","d2","d3","d4","d5","d6","d7",\
  "r0","r1","r2","r3","cc","memory");\
  float *c_tmp = c_ptr;\
  NEON_SGEMM_SAVE_M##mdim##N##ndim##_ASM\
}

static inline void inline_dualpack_gemm_afloat_bfloat_cfloat_m6_n8(
  const float *a_head, const float *b_head, float *c_ptr,
  uint32_t K, float beta, uint32_t ldc) {
  NEON_SGEMM_ASM(6, 8, A53)
}

static inline void inline_dualpack_gemm_afloat_bfloat_cfloat_m8_n6(
  const float *a_head, const float *b_head, float *c_ptr,
  uint32_t K, float beta, uint32_t ldc) {
  NEON_SGEMM_ASM(8, 6, A53)
}

DUALPACK_KERNEL_FUNC_LM(sgemm, float, float, float, 6, 8)
DUALPACK_KERNEL_FUNC_LN(sgemm, float, float, float, 8, 6)

