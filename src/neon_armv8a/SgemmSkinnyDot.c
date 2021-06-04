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


#define _GNU_SOURCE
#include "arm_neon/ARMCompareAndSwap.h"
#include "arm_neon/ARMCpuType.h"
#include "common/CommonSkinnyDot.h"
#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotA35.h"
#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotA53.h"
#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotA7x.h"
#include <arm_neon.h>
#include <sched.h>

typedef float sgemm_skinnydot_ascalar;
typedef float sgemm_skinnydot_bscalar;
typedef float sgemm_skinnydot_cscalar;

static inline void inline_sgemm_arowmajor_bskinny_m4n1(const float *a_ptr1,
  const float *b_ptr, float *c_ptr, uint32_t k_inc, uint32_t LDK,
  uint32_t LDM, float beta, bool c_rowmajor) {

  const float *a_ptr2 = a_ptr1 + LDK;
  const float *a_ptr3 = a_ptr1 + LDK * 2;
  const float *a_ptr4 = a_ptr2 + LDK * 2;

  float32x2_t cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8;
  const float *a_pref = a_ptr4 + LDK;
  const uint32_t pref_inc = (LDK > k_inc) ?
    (LDK - k_inc) * sizeof(float) : 64;
  uint32_t k_left = k_inc;
  __asm__ __volatile__(
    "movz w0,#0; movz w1,#64\n\t" //pref
    "movi %[cd1].8b,#0; movi %[cd2].8b,#0\n\t"
    "movi %[cd3].8b,#0; movi %[cd4].8b,#0\n\t"
    "movi %[cd5].8b,#0; movi %[cd6].8b,#0\n\t"
    "movi %[cd7].8b,#0; movi %[cd8].8b,#0\n\t"
    "cmp %w[k_left],#4; b.lt 3f\n\t"
    "ldr d2,[%[a_ptr1]],#16; ldr d3,[%[a_ptr2]],#16\n\t"
    "ldr d4,[%[a_ptr3]],#16; ldr d5,[%[a_ptr4]],#16\n\t"
    "ldr d0,[%[b_ptr]],#16\n\t"
    "ldr d6,[%[a_ptr1],#-8]; ldr d7,[%[a_ptr2],#-8]\n\t"
    "ldr d8,[%[a_ptr3],#-8]; ldr d9,[%[a_ptr4],#-8]\n\t"
    "ldr d1,[%[b_ptr],#-8]\n\t"
    "cmp %w[k_left],#8; b.lt 2f\n\t"
    ".balign 16; 1:\n\t"
    "prfm pldl2keep,[%[a_pref]]; add w0,w0,#16\n\t"
    "fmla %[cd1].2s,v2.2s,v0.2s; ldr d2,[%[a_ptr1]],#16\n\t"
    "cmp w0,%w[k_inc]\n\t"
    "fmla %[cd2].2s,v3.2s,v0.2s; ldr d3,[%[a_ptr2]],#16\n\t"
    "csel w2,%w[pref_inc],w1,gt\n\t"
    "fmla %[cd3].2s,v4.2s,v0.2s; ldr d4,[%[a_ptr3]],#16\n\t"
    "fmla %[cd4].2s,v5.2s,v0.2s; ldr d5,[%[a_ptr4]],#16\n\t"
    "csel w0,wzr,w0,gt\n\t"
    "ldr d0,[%[b_ptr]],#16; sub %w[k_left],%w[k_left],#4\n\t"
    "fmla %[cd5].2s,v6.2s,v1.2s; ldr d6,[%[a_ptr1],#-8]\n\t"
    "add %[a_pref],%[a_pref],x2\n\t"
    "fmla %[cd6].2s,v7.2s,v1.2s; ldr d7,[%[a_ptr2],#-8]\n\t"
    "cmp %w[k_left],#8\n\t"
    "fmla %[cd7].2s,v8.2s,v1.2s; ldr d8,[%[a_ptr3],#-8]\n\t"
    "fmla %[cd8].2s,v9.2s,v1.2s; ldr d9,[%[a_ptr4],#-8]\n\t"
    "ldr d1,[%[b_ptr],#-8]; b.ge 1b\n\t"
    "2:\n\t"
    "fmla %[cd1].2s,v2.2s,v0.2s; fmla %[cd2].2s,v3.2s,v0.2s\n\t"
    "fmla %[cd3].2s,v4.2s,v0.2s; fmla %[cd4].2s,v5.2s,v0.2s\n\t"
    "sub %w[k_left],%w[k_left],#4\n\t"
    "fmla %[cd5].2s,v6.2s,v1.2s; fmla %[cd6].2s,v7.2s,v1.2s\n\t"
    "fmla %[cd7].2s,v8.2s,v1.2s; fmla %[cd8].2s,v9.2s,v1.2s\n\t"
    "3:\n\t"
   :[cd1]"=w"(cd1), [cd2]"=w"(cd2), [cd3]"=w"(cd3), [cd4]"=w"(cd4),
    [cd5]"=w"(cd5), [cd6]"=w"(cd6), [cd7]"=w"(cd7), [cd8]"=w"(cd8),
    [k_left]"+r"(k_left), [a_pref]"+r"(a_pref), [b_ptr]"+r"(b_ptr),
    [a_ptr1]"+r"(a_ptr1), [a_ptr2]"+r"(a_ptr2),
    [a_ptr3]"+r"(a_ptr3), [a_ptr4]"+r"(a_ptr4)
   :[k_inc]"r"(k_inc), [pref_inc]"r"(pref_inc)
   :"cc","memory","x0","x1","x2",
    "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9");

  cd1 = vadd_f32(cd1, cd5); cd2 = vadd_f32(cd2, cd6);
  cd3 = vadd_f32(cd3, cd7); cd4 = vadd_f32(cd4, cd8);
  float cs1 = vpadds_f32(cd1);
  float cs2 = vpadds_f32(cd2);
  float cs3 = vpadds_f32(cd3);
  float cs4 = vpadds_f32(cd4);

  for (; k_left > 0; k_left--) {
    float bs1 = *b_ptr; b_ptr++;
    cs1 += (*a_ptr1) * bs1; a_ptr1++; 
    cs2 += (*a_ptr2) * bs1; a_ptr2++;
    cs3 += (*a_ptr3) * bs1; a_ptr3++; 
    cs4 += (*a_ptr4) * bs1; a_ptr4++;
  }
  c_ptr[0] = c_ptr[0] * beta + cs1;
  c_ptr[1] = c_ptr[1] * beta + cs2;
  c_ptr[2] = c_ptr[2] * beta + cs3;
  c_ptr[3] = c_ptr[3] * beta + cs4;
}

static inline void inline_sgemm_arowmajor_bskinny_m1n1(const float *a_ptr,
  const float *b_ptr, float *c_ptr, uint32_t k_inc, uint32_t LDK,
  uint32_t LDM, float beta, bool c_rowmajor) {

  float cs1;
  __asm__ __volatile__(
    "movi v16.8b,#0; movi v17.8b,#0\n\t"
    "mov v18.8b,v16.8b; mov v19.8b,v17.8b\n\t"
    "mov v20.8b,v16.8b; mov v21.8b,v17.8b\n\t"
    "mov v22.8b,v16.8b; mov v23.8b,v17.8b\n\t"
    "cmp %w[K],#16; b.lt 4f\n\t"
    "prfm pldl1keep,[%[a_ptr],#256]\n\t"
    "ldr d0,[%[a_ptr]],#64; ldr d8,[%[b_ptr]],#64\n\t"
    "ldr d1,[%[a_ptr],#-56]; ldr d9,[%[b_ptr],#-56]\n\t"
    "ldr d2,[%[a_ptr],#-48]; ldr d10,[%[b_ptr],#-48]\n\t"
    "ldr d3,[%[a_ptr],#-40]; ldr d11,[%[b_ptr],#-40]\n\t"
    "ldr d4,[%[a_ptr],#-32]; ldr d12,[%[b_ptr],#-32]\n\t"
    "ldr d5,[%[a_ptr],#-24]; ldr d13,[%[b_ptr],#-24]\n\t"
    "ldr d6,[%[a_ptr],#-16]; ldr d14,[%[b_ptr],#-16]\n\t"
    "ldr d7,[%[a_ptr],#-8]; ldr d15,[%[b_ptr],#-8]\n\t"
    "cmp %w[K],#32; b.lt 3f\n\t"
    "2:\n\t"
    "prfm pldl1keep,[%[a_ptr],#256]\n\t"
    "fmla v16.2s,v0.2s,v8.2s; ldr d0,[%[a_ptr]],#64; ldr d8,[%[b_ptr]],#64\n\t"
    "fmla v17.2s,v1.2s,v9.2s; ldr d1,[%[a_ptr],#-56]; ldr d9,[%[b_ptr],#-56]\n\t"
    "fmla v18.2s,v2.2s,v10.2s; ldr d2,[%[a_ptr],#-48]; ldr d10,[%[b_ptr],#-48]\n\t"
    "fmla v19.2s,v3.2s,v11.2s; ldr d3,[%[a_ptr],#-40]; ldr d11,[%[b_ptr],#-40]\n\t"
    "sub %w[K],%w[K],#16\n\t"
    "fmla v20.2s,v4.2s,v12.2s; ldr d4,[%[a_ptr],#-32]; ldr d12,[%[b_ptr],#-32]\n\t"
    "fmla v21.2s,v5.2s,v13.2s; ldr d5,[%[a_ptr],#-24]; ldr d13,[%[b_ptr],#-24]\n\t"
    "cmp %w[K],#32\n\t"
    "fmla v22.2s,v6.2s,v14.2s; ldr d6,[%[a_ptr],#-16]; ldr d14,[%[b_ptr],#-16]\n\t"
    "fmla v23.2s,v7.2s,v15.2s; ldr d7,[%[a_ptr],#-8]; ldr d15,[%[b_ptr],#-8]\n\t"
    "b.ge 2b\n\t"
    "3:\n\t"
    "fmla v16.2s,v0.2s,v8.2s; fmla v17.2s,v1.2s,v9.2s\n\t"
    "fmla v18.2s,v2.2s,v10.2s; fmla v19.2s,v3.2s,v11.2s; sub %w[K],%w[K],#16\n\t"
    "fmla v20.2s,v4.2s,v12.2s; fmla v21.2s,v5.2s,v13.2s\n\t"
    "fmla v22.2s,v6.2s,v14.2s; fmla v23.2s,v7.2s,v15.2s\n\t"
    "4:\n\t"
    "fadd v16.2s,v16.2s,v20.2s; fadd v17.2s,v17.2s,v21.2s\n\t"
    "fadd v18.2s,v18.2s,v22.2s; fadd v19.2s,v19.2s,v23.2s\n\t"
    "cmp %w[K],#8; b.lt 5f\n\t"
    "ldr d0,[%[a_ptr]],#32; ldr d8,[%[b_ptr]],#32; fmla v16.2s,v0.2s,v8.2s\n\t"
    "ldr d1,[%[a_ptr],#-24]; ldr d9,[%[b_ptr],#-24]; fmla v17.2s,v1.2s,v9.2s\n\t"
    "sub %w[K],%w[K],#8\n\t"
    "ldr d2,[%[a_ptr],#-16]; ldr d10,[%[b_ptr],#-16]; fmla v18.2s,v2.2s,v10.2s\n\t"
    "ldr d3,[%[a_ptr],#-8]; ldr d11,[%[b_ptr],#-8]; fmla v19.2s,v3.2s,v11.2s\n\t"
    "5:\n\t"
    "fadd v16.2s,v16.2s,v18.2s; fadd v17.2s,v17.2s,v19.2s\n\t"
    "cmp %w[K],#4; b.lt 6f\n\t"
    "ldr d0,[%[a_ptr]],#16; ldr d8,[%[b_ptr]],#16; fmla v16.2s,v0.2s,v8.2s\n\t"
    "sub %w[K],%w[K],#4\n\t"
    "ldr d1,[%[a_ptr],#-8]; ldr d9,[%[b_ptr],#-8]; fmla v17.2s,v1.2s,v9.2s\n\t"
    "6:\n\t"
    "fadd v16.2s,v16.2s,v17.2s\n\t"
    "cmp %w[K],#2; b.lt 7f\n\t"
    "ldr d0,[%[a_ptr]],#8; ldr d8,[%[b_ptr]],#8; fmla v16.2s,v0.2s,v8.2s\n\t"
    "sub %w[K],%w[K],#2\n\t"
    "7:\n\t"
    "faddp %s[cs1],v16.2s\n\t"
    "cmp %w[K],#1; b.lt 10f\n\t"
    "ldr s0,[%[a_ptr]],#4; ldr s8,[%[b_ptr]],#4; fmla %s[cs1],s0,v8.s[0]\n\t"
    "10:\n\t"
   :[cs1]"=w"(cs1), [a_ptr]"+r"(a_ptr), [b_ptr]"+r"(b_ptr), [K]"+r"(k_inc)
   ::"cc","memory","v0","v1","v2","v3","v4","v5",
     "v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17",
     "v18","v19","v20","v21","v22","v23");
   c_ptr[0] = c_ptr[0] * beta + cs1;
}

/* k_mask = 7 */
static inline void inline_sgemm_arowmajor_bskinny_m4n2(const float *a_ptr1,
  const float *b_ptr, float *c_ptr, uint32_t k_inc, uint32_t LDK,
  uint32_t LDM, float beta, bool c_rowmajor) {

  float32x4_t cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8;
  const float *a_ptr2 = a_ptr1 + LDK;
  const float *a_ptr3 = a_ptr1 + LDK * 2;
  const float *a_ptr4 = a_ptr2 + LDK * 2;
  uint32_t k_left = k_inc;
  const float *a_pref = a_ptr4 + LDK;
  const uint32_t pref_inc = (LDK > k_inc) ?
    (LDK - k_inc) * sizeof(float) : 64;
  __asm__ __volatile__(
    "movz w0,#0; movz w1,#64\n\t" //pref
    "movi %[cq1].16b,#0; movi %[cq2].16b,#0\n\t"
    "movi %[cq3].16b,#0; movi %[cq4].16b,#0\n\t"
    "movi %[cq5].16b,#0; movi %[cq6].16b,#0\n\t"
    "movi %[cq7].16b,#0; movi %[cq8].16b,#0\n\t"
    "cmp %w[k_left],#4; b.lt 3f\n\t"
    "ldr q2,[%[a_ptr1]],#16; ldr q3,[%[a_ptr2]],#16\n\t"
    "ldr q4,[%[a_ptr3]],#16; ldr q5,[%[a_ptr4]],#16\n\t"
    "ldr q0,[%[b_ptr]]; ldr q1,[%[b_ptr],#16]; add %[b_ptr],%[b_ptr],#32\n\t"
    "cmp %w[k_left],#8; b.lt 2f\n\t"
    ".balign 16; 1:\n\t"
    "prfm pldl2keep,[%[a_pref]]; add w0,w0,#16\n\t"
    "fmla %[cq1].4s,v2.4s,v0.4s; fmla %[cq5].4s,v2.4s,v1.4s\n\t"
    "ldr q2,[%[a_ptr1]],#16; cmp w0,%w[k_inc]\n\t"
    "fmla %[cq2].4s,v3.4s,v0.4s; fmla %[cq6].4s,v3.4s,v1.4s\n\t"
    "ldr q3,[%[a_ptr2]],#16; csel w2,%w[pref_inc],w1,gt\n\t"
    "sub %w[k_left],%w[k_left],#4\n\t"
    "fmla %[cq3].4s,v4.4s,v0.4s; fmla %[cq7].4s,v4.4s,v1.4s\n\t"
    "ldr q4,[%[a_ptr3]],#16; csel w0,wzr,w0,gt\n\t"
    "cmp %w[k_left],#8\n\t"
    "fmla %[cq4].4s,v5.4s,v0.4s; fmla %[cq8].4s,v5.4s,v1.4s\n\t"
    "ldr q5,[%[a_ptr4]],#16; add %[a_pref],%[a_pref],x2\n\t"
    "ldr q0,[%[b_ptr]]; ldr q1,[%[b_ptr],#16]\n\t"
    "add %[b_ptr],%[b_ptr],#32; b.ge 1b\n\t"
    "2:\n\t"
    "fmla %[cq1].4s,v2.4s,v0.4s; fmla %[cq5].4s,v2.4s,v1.4s\n\t"
    "fmla %[cq2].4s,v3.4s,v0.4s; fmla %[cq6].4s,v3.4s,v1.4s\n\t"
    "fmla %[cq3].4s,v4.4s,v0.4s; fmla %[cq7].4s,v4.4s,v1.4s\n\t"
    "fmla %[cq4].4s,v5.4s,v0.4s; fmla %[cq8].4s,v5.4s,v1.4s\n\t"
    "sub %w[k_left],%w[k_left],#4\n\t"
    "3:\n\t"
   :[cq1]"=w"(cq1), [cq2]"=w"(cq2), [cq3]"=w"(cq3), [cq4]"=w"(cq4),
    [cq5]"=w"(cq5), [cq6]"=w"(cq6), [cq7]"=w"(cq7), [cq8]"=w"(cq8),
    [k_left]"+r"(k_left), [a_pref]"+r"(a_pref),
    [b_ptr]"+r"(b_ptr), [a_ptr1]"+r"(a_ptr1), [a_ptr2]"+r"(a_ptr2),
    [a_ptr3]"+r"(a_ptr3), [a_ptr4]"+r"(a_ptr4)
   :[k_inc]"r"(k_inc), [pref_inc]"r"(pref_inc)
   :"x0","x1","x2","v0","v1","v2","v3","v4","v5","cc","memory");

  cq1 = vpaddq_f32(cq1, cq5);
  cq2 = vpaddq_f32(cq2, cq6);
  cq3 = vpaddq_f32(cq3, cq7);
  cq4 = vpaddq_f32(cq4, cq8);
  
  if (k_left >= 2) {
    float32x4_t bq1 = vld1q_f32(b_ptr); b_ptr += 4;
    float32x2_t ad1 = vld1_f32(a_ptr1); a_ptr1 += 2;
    float32x2_t ad2 = vld1_f32(a_ptr2); a_ptr2 += 2;
    float32x2_t ad3 = vld1_f32(a_ptr3); a_ptr3 += 2;
    float32x2_t ad4 = vld1_f32(a_ptr4); a_ptr4 += 2;
    float32x4_t aq1 = vcombine_f32(ad1, ad1);
    float32x4_t aq2 = vcombine_f32(ad2, ad2);
    float32x4_t aq3 = vcombine_f32(ad3, ad3);
    float32x4_t aq4 = vcombine_f32(ad4, ad4);
    cq1 = vfmaq_f32(cq1, aq1, bq1);
    cq2 = vfmaq_f32(cq2, aq2, bq1);
    cq3 = vfmaq_f32(cq3, aq3, bq1);
    cq4 = vfmaq_f32(cq4, aq4, bq1);
    k_left -= 2;
  }
  
  float32x2_t cd1 = vget_low_f32(vpaddq_f32(cq1, cq1));
  float32x2_t cd2 = vget_low_f32(vpaddq_f32(cq2, cq2));
  float32x2_t cd3 = vget_low_f32(vpaddq_f32(cq3, cq3));
  float32x2_t cd4 = vget_low_f32(vpaddq_f32(cq4, cq4));
  
  if (k_left > 0) {
    float32x2_t bd1 = vld1_f32(b_ptr);
    float32x2_t ad1 = vld1_dup_f32(a_ptr1);
    float32x2_t ad2 = vld1_dup_f32(a_ptr2);
    float32x2_t ad3 = vld1_dup_f32(a_ptr3);
    float32x2_t ad4 = vld1_dup_f32(a_ptr4);
    cd1 = vfma_f32(cd1, ad1, bd1);
    cd2 = vfma_f32(cd2, ad2, bd1);
    cd3 = vfma_f32(cd3, ad3, bd1);
    cd4 = vfma_f32(cd4, ad4, bd1);
  }
  
  if (c_rowmajor) {
    cd1 = vfma_n_f32(cd1, vld1_f32(c_ptr), beta);
    cd2 = vfma_n_f32(cd2, vld1_f32(c_ptr + 2), beta);
    cd3 = vfma_n_f32(cd3, vld1_f32(c_ptr + 4), beta);
    cd4 = vfma_n_f32(cd4, vld1_f32(c_ptr + 6), beta);
    vst1_f32(c_ptr, cd1);
    vst1_f32(c_ptr + 2, cd2);
    vst1_f32(c_ptr + 4, cd3);
    vst1_f32(c_ptr + 6, cd4);
  } else {
    float32x2_t cd00 = vzip1_f32(cd1, cd2);
    float32x2_t cd01 = vzip1_f32(cd3, cd4);
    float32x2_t cd10 = vzip2_f32(cd1, cd2);
    float32x2_t cd11 = vzip2_f32(cd3, cd4);
    float *c_ptr1 = c_ptr;
    float *c_ptr2 = c_ptr + LDM;
    cd00 = vfma_n_f32(cd00, vld1_f32(c_ptr1), beta);
    cd01 = vfma_n_f32(cd01, vld1_f32(c_ptr1 + 2), beta);
    cd10 = vfma_n_f32(cd10, vld1_f32(c_ptr2), beta);
    cd11 = vfma_n_f32(cd11, vld1_f32(c_ptr2 + 2), beta);
    vst1_f32(c_ptr1, cd00);
    vst1_f32(c_ptr1 + 2, cd01);
    vst1_f32(c_ptr2, cd10);
    vst1_f32(c_ptr2 + 2, cd11);
  }
}

static inline void inline_sgemm_arowmajor_bskinny_m1n2(const float *a_ptr,
  const float *b_ptr, float *c_ptr, uint32_t k_inc, uint32_t LDK,
  uint32_t LDM, float beta, bool c_rowmajor) {

  uint32_t k_left = k_inc;
  float cs1, cs2;
  __asm__ __volatile__ (
    "movi v8.16b,#0; movi v9.16b,#0\n\t"
    "mov v10.16b,v8.16b; mov v11.16b,v9.16b\n\t"
    "mov v16.16b,v8.16b; mov v17.16b,v9.16b\n\t"
    "mov v18.16b,v8.16b; mov v19.16b,v9.16b\n\t"
    "cmp %w[k_left],#16; b.lt 4f\n\t"
    "prfm pldl1keep,[%[a_ptr],#256]\n\t"
    "ldr q0,[%[a_ptr]]; ldr q1,[%[a_ptr],#16]\n\t"
    "ldr q2,[%[a_ptr],#32]; ldr q3,[%[a_ptr],#48]\n\t"
    "add %[a_ptr],%[a_ptr],#64\n\t"
    "ldr q4,[%[b_ptr]]; ldr q12,[%[b_ptr],#16]\n\t"
    "ldr q5,[%[b_ptr],#32]; ldr q13,[%[b_ptr],#48]\n\t"
    "ldr q6,[%[b_ptr],#64]; ldr q14,[%[b_ptr],#80]\n\t"
    "ldr q7,[%[b_ptr],#96]; ldr q15,[%[b_ptr],#112]\n\t"
    "add %[b_ptr],%[b_ptr],#128\n\t"
    "cmp %w[k_left],#32; b.lt 3f\n\t"
    ".balign 16; 2:\n\t"
    "prfm pldl1keep,[%[a_ptr],#256]\n\t"
    "fmla v8.4s,v0.4s,v4.4s; ldr q4,[%[b_ptr]]\n\t"
    "fmla v10.4s,v0.4s,v12.4s; ldr q12,[%[b_ptr],#16]; ldr q0,[%[a_ptr]],#64\n\t"
    "fmla v9.4s,v1.4s,v5.4s; ldr q5,[%[b_ptr],#32]\n\t"
    "fmla v11.4s,v1.4s,v13.4s; ldr q13,[%[b_ptr],#48]; ldr q1,[%[a_ptr],#-48]\n\t"
    "sub %w[k_left],%w[k_left],#16\n\t"
    "fmla v16.4s,v2.4s,v6.4s; ldr q6,[%[b_ptr],#64]\n\t"
    "fmla v18.4s,v2.4s,v14.4s; ldr q14,[%[b_ptr],#80]; ldr q2,[%[a_ptr],#-32]\n\t"
    "cmp %w[k_left],#32\n\t"
    "fmla v17.4s,v3.4s,v7.4s; ldr q7,[%[b_ptr],#96]\n\t"
    "fmla v19.4s,v3.4s,v15.4s; ldr q15,[%[b_ptr],#112]; ldr q3,[%[a_ptr],#-16]\n\t"
    "add %[b_ptr],%[b_ptr],#128; b.ge 2b\n\t"
    "3:\n\t"
    "fmla v8.4s,v0.4s,v4.4s; fmla v10.4s,v0.4s,v12.4s\n\t"
    "fmla v9.4s,v1.4s,v5.4s; fmla v11.4s,v1.4s,v13.4s\n\t"
    "sub %w[k_left],%w[k_left],#16\n\t"
    "fmla v16.4s,v2.4s,v6.4s; fmla v18.4s,v2.4s,v14.4s\n\t"
    "fmla v17.4s,v3.4s,v7.4s; fmla v19.4s,v3.4s,v15.4s\n\t"
    "4:\n\t"
    "fadd v8.4s,v8.4s,v16.4s; fadd v9.4s,v9.4s,v17.4s\n\t"
    "fadd v10.4s,v10.4s,v18.4s; fadd v11.4s,v11.4s,v19.4s\n\t"
    "cmp %w[k_left],#8; b.lt 5f\n\t"
    "ldr q0,[%[a_ptr]],#32; ldr q4,[%[b_ptr]]; ldr q12,[%[b_ptr],#16]\n\t"
    "fmla v8.4s,v0.4s,v4.4s; fmla v10.4s,v0.4s,v12.4s\n\t"
    "sub %w[k_left],%w[k_left],#8\n\t"
    "ldr q1,[%[a_ptr],#-16]; ldr q5,[%[b_ptr],#32]; ldr q13,[%[b_ptr],#48]\n\t"
    "add %[b_ptr],%[b_ptr],#64\n\t"
    "fmla v9.4s,v1.4s,v5.4s; fmla v11.4s,v1.4s,v13.4s\n\t"
    "5:\n\t"
    "fadd v8.4s,v8.4s,v9.4s; fadd v10.4s,v10.4s,v11.4s\n\t"
    "cmp %w[k_left],#4; b.lt 6f\n\t"
    "ldr q0,[%[a_ptr]],#16; ldr q4,[%[b_ptr]]; ldr q12,[%[b_ptr],#16]\n\t"
    "fmla v8.4s,v0.4s,v4.4s; fmla v10.4s,v0.4s,v12.4s\n\t"
    "add %[b_ptr],%[b_ptr],#32; sub %w[k_left],%w[k_left],#4\n\t"
    "6:\n\t"
    "movi v9.16b,#0; faddp v8.4s,v8.4s,v9.4s; faddp v10.4s,v10.4s,v9.4s\n\t"
    "cmp %w[k_left],#2; b.lt 7f\n\t"
    "ldr d0,[%[a_ptr]],#8; ldr d4,[%[b_ptr]]; ldr d12,[%[b_ptr],#8]\n\t"
    "fmla v8.2s,v0.2s,v4.2s; fmla v10.2s,v0.2s,v12.2s\n\t"
    "add %[b_ptr],%[b_ptr],#16; sub %w[k_left],%w[k_left],#2\n\t"
    "7:\n\t"
    "faddp %s[cs1],v8.2s; faddp %s[cs2],v10.2s\n\t"
    "cmp %w[k_left],#1; b.lt 10f\n\t"
    "ldr s0,[%[a_ptr]],#4; ldr s4,[%[b_ptr]]; ldr s12,[%[b_ptr],#4]\n\t"
    "fmla %s[cs1],s0,v4.s[0]; fmla %s[cs2],s0,v12.s[0]\n\t"
    "10:\n\t"
   :[cs1]"=w"(cs1), [cs2]"=w"(cs2),
    [a_ptr]"+r"(a_ptr), [b_ptr]"+r"(b_ptr), [k_left]"+r"(k_left)
   ::"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7",
     "v8","v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19");

  if (c_rowmajor) {
    c_ptr[0] = c_ptr[0] * beta + cs1;
    c_ptr[1] = c_ptr[1] * beta + cs2;
  } else {
    c_ptr[0] = c_ptr[0] * beta + cs1;
    c_ptr[LDM] = c_ptr[LDM] * beta + cs2;
  }
}

/* k_mask = 7 */
static inline void inline_sgemm_arowmajor_bskinny_m4n3(const float *a_ptr1,
  const float *b_ptr, float *c_ptr, uint32_t k_inc, uint32_t LDK,
  uint32_t LDM, float beta, bool c_rowmajor) {

  const float *a_ptr2 = a_ptr1 + LDK;
  const float *a_ptr3 = a_ptr1 + LDK * 2;
  const float *a_ptr4 = a_ptr2 + LDK * 2;
  uint32_t k_left = k_inc;
  uint32_t next_pref = (LDK * 4 >= k_inc) ?
    (LDK * 4 - k_inc + 4) * sizeof(float) : 64;
  float32x4_t cq1, cq2, cq3;
  __asm__ __volatile__(
    "movi %[q1].16b,#0; movi %[q2].16b,#0; movi %[q3].16b,#0\n\t"
    "movi v10.16b,#0; movi v11.16b,#0; movi v12.16b,#0\n\t"
    "movi v13.16b,#0; movi v14.16b,#0; movi v15.16b,#0\n\t"
    "movi v16.16b,#0; movi v17.16b,#0; movi v18.16b,#0\n\t"
    "cmp %w[k_left],#4; b.lt 4f\n\t"
    "ldr q0,[%[a_ptr1]],#16; ldr q1,[%[a_ptr2]],#16\n\t"
    "ldr q2,[%[a_ptr3]],#16; ldr q3,[%[a_ptr4]],#16\n\t"
    "ldr q4,[%[b_ptr]]; ldr q5,[%[b_ptr],#16]\n\t"
    "ldr q6,[%[b_ptr],#32]; add %[b_ptr],%[b_ptr],#48\n\t"
    "cmp %w[k_left],#12; b.lt 2f\n\t"
    ".balign 16; 1:\n\t"
    "fmla %[q1].4s,v0.4s,v4.4s; ldr q7,[%[b_ptr]],#96\n\t"
    "fmla %[q2].4s,v0.4s,v5.4s\n\t"
    "fmla %[q3].4s,v0.4s,v6.4s; ldr q0,[%[a_ptr1]],#32\n\t"
    "fmla v10.4s,v1.4s,v4.4s; ldr q8,[%[b_ptr],#-80]\n\t"
    "fmla v11.4s,v1.4s,v5.4s; prfm pldl1keep,[%[a_ptr1],#64]\n\t"
    "fmla v12.4s,v1.4s,v6.4s; ldr q1,[%[a_ptr2]],#32\n\t"
    "fmla v13.4s,v2.4s,v4.4s; ldr q9,[%[b_ptr],#-64]\n\t"
    "fmla v14.4s,v2.4s,v5.4s; prfm pldl1keep,[%[a_ptr2],#64]\n\t"
    "fmla v15.4s,v2.4s,v6.4s; ldr q2,[%[a_ptr3]],#32\n\t"
    "fmla v16.4s,v3.4s,v4.4s\n\t"
    "fmla v17.4s,v3.4s,v5.4s; prfm pldl1keep,[%[a_ptr3],#64]\n\t"
    "fmla v18.4s,v3.4s,v6.4s; ldr q3,[%[a_ptr4]],#32\n\t"
    "fmla %[q1].4s,v0.4s,v7.4s; ldr q4,[%[b_ptr],#-48]\n\t"
    "fmla %[q2].4s,v0.4s,v8.4s; prfm pldl1keep,[%[a_ptr4],#64]\n\t"
    "fmla %[q3].4s,v0.4s,v9.4s; ldr q0,[%[a_ptr1],#-16]\n\t"
    "fmla v10.4s,v1.4s,v7.4s; ldr q5,[%[b_ptr],#-32]\n\t"
    "fmla v11.4s,v1.4s,v8.4s\n\t"
    "fmla v12.4s,v1.4s,v9.4s; ldr q1,[%[a_ptr2],#-16]\n\t"
    "fmla v13.4s,v2.4s,v7.4s; ldr q6,[%[b_ptr],#-16]\n\t"
    "fmla v14.4s,v2.4s,v8.4s; sub %w[k_left],%w[k_left],#8\n\t"
    "fmla v15.4s,v2.4s,v9.4s; ldr q2,[%[a_ptr3],#-16]\n\t"
    "fmla v16.4s,v3.4s,v7.4s; cmp %w[k_left],#12\n\t"
    "fmla v17.4s,v3.4s,v8.4s\n\t"
    "fmla v18.4s,v3.4s,v9.4s; ldr q3,[%[a_ptr4],#-16]; b.ge 1b\n\t"
    "2:\n\t"
    "cmp %w[k_left],#8; b.lt 3f\n\t"
    "fmla %[q1].4s,v0.4s,v4.4s; ldr q7,[%[b_ptr]],#48\n\t"
    "fmla %[q2].4s,v0.4s,v5.4s\n\t"
    "fmla %[q3].4s,v0.4s,v6.4s; ldr q0,[%[a_ptr1]],#16\n\t"
    "fmla v10.4s,v1.4s,v4.4s; ldr q8,[%[b_ptr],#-32]\n\t"
    "fmla v11.4s,v1.4s,v5.4s\n\t"
    "prfm pldl1keep,[%[a_ptr1],%w[next_pref],SXTW #0]\n\t"
    "fmla v12.4s,v1.4s,v6.4s; ldr q1,[%[a_ptr2]],#16\n\t"
    "fmla v13.4s,v2.4s,v4.4s; ldr q9,[%[b_ptr],#-16]\n\t"
    "fmla v14.4s,v2.4s,v5.4s\n\t"
    "prfm pldl1keep,[%[a_ptr2],%w[next_pref],SXTW #0]\n\t"
    "fmla v15.4s,v2.4s,v6.4s; ldr q2,[%[a_ptr3]],#16\n\t"
    "fmla v16.4s,v3.4s,v4.4s\n\t"
    "fmla v17.4s,v3.4s,v5.4s\n\t"
    "prfm pldl1keep,[%[a_ptr3],%w[next_pref],SXTW #0]\n\t"
    "fmla v18.4s,v3.4s,v6.4s; ldr q3,[%[a_ptr4]],#16\n\t"
    "fmla %[q1].4s,v0.4s,v7.4s\n\t"
    "fmla %[q2].4s,v0.4s,v8.4s\n\t"
    "prfm pldl1keep,[%[a_ptr4],%w[next_pref],SXTW #0]\n\t"
    "fmla %[q3].4s,v0.4s,v9.4s\n\t"
    "fmla v10.4s,v1.4s,v7.4s\n\t"
    "fmla v11.4s,v1.4s,v8.4s\n\t"
    "fmla v12.4s,v1.4s,v9.4s\n\t"
    "fmla v13.4s,v2.4s,v7.4s\n\t"
    "fmla v14.4s,v2.4s,v8.4s; sub %w[k_left],%w[k_left],#8\n\t"
    "fmla v15.4s,v2.4s,v9.4s\n\t"
    "fmla v16.4s,v3.4s,v7.4s\n\t"
    "fmla v17.4s,v3.4s,v8.4s\n\t"
    "fmla v18.4s,v3.4s,v9.4s; b 4f\n\t"
    "3:\n\t"
    "fmla %[q1].4s,v0.4s,v4.4s\n\t"
    "fmla %[q2].4s,v0.4s,v5.4s\n\t"
    "prfm pldl1keep,[%[a_ptr1],%w[next_pref],SXTW #0]\n\t"
    "fmla %[q3].4s,v0.4s,v6.4s\n\t"
    "fmla v10.4s,v1.4s,v4.4s\n\t"
    "prfm pldl1keep,[%[a_ptr2],%w[next_pref],SXTW #0]\n\t"
    "fmla v11.4s,v1.4s,v5.4s\n\t"
    "fmla v12.4s,v1.4s,v6.4s\n\t"
    "prfm pldl1keep,[%[a_ptr3],%w[next_pref],SXTW #0]\n\t"
    "fmla v13.4s,v2.4s,v4.4s\n\t"
    "fmla v14.4s,v2.4s,v5.4s; sub %w[k_left],%w[k_left],#4\n\t"
    "prfm pldl1keep,[%[a_ptr4],%w[next_pref],SXTW #0]\n\t"
    "fmla v15.4s,v2.4s,v6.4s\n\t"
    "fmla v16.4s,v3.4s,v4.4s\n\t"
    "fmla v17.4s,v3.4s,v5.4s\n\t"
    "fmla v18.4s,v3.4s,v6.4s\n\t"
    "4:\n\t"
    "faddp %[q1].4s,%[q1].4s,v10.4s; faddp v13.4s,v13.4s,v16.4s\n\t"
    "faddp %[q2].4s,%[q2].4s,v11.4s; faddp v14.4s,v14.4s,v17.4s\n\t"
    "faddp %[q3].4s,%[q3].4s,v12.4s; faddp v15.4s,v15.4s,v18.4s\n\t"
    "cmp %w[k_left],#2; b.lt 5f\n\t"
    "ldr d0,[%[a_ptr1]],#8; ldr d1,[%[a_ptr2]],#8\n\t"
    "ldr d2,[%[a_ptr3]],#8; ldr d3,[%[a_ptr4]],#8\n\t"
    "ld1r {v4.2d},[%[b_ptr]],#8; ins v0.d[1],v1.d[0]\n\t"
    "ld1r {v5.2d},[%[b_ptr]],#8; ins v2.d[1],v3.d[0]\n\t"
    "ld1r {v6.2d},[%[b_ptr]],#8; sub %w[k_left],%w[k_left],#2\n\t"
    "fmla %[q1].4s,v0.4s,v4.4s\n\t"
    "fmla %[q2].4s,v0.4s,v5.4s\n\t"
    "fmla %[q3].4s,v0.4s,v6.4s\n\t"
    "fmla v13.4s,v2.4s,v4.4s\n\t"
    "fmla v14.4s,v2.4s,v5.4s\n\t"
    "fmla v15.4s,v2.4s,v6.4s\n\t"
    "5:\n\t"
    "faddp %[q1].4s,%[q1].4s,v13.4s\n\t"
    "faddp %[q2].4s,%[q2].4s,v14.4s\n\t"
    "faddp %[q3].4s,%[q3].4s,v15.4s\n\t"
    "cmp %w[k_left],#1; b.lt 6f\n\t"
    "ldr s0,[%[a_ptr1]],#4; ldr s1,[%[a_ptr2]],#4\n\t"
    "ldr s2,[%[a_ptr3]],#4; ldr s3,[%[a_ptr4]],#4\n\t"
    "ldr s4,[%[b_ptr]],#4; ins v0.s[1],v1.s[0]\n\t"
    "ldr s5,[%[b_ptr]],#4; ins v2.s[1],v3.s[0]\n\t"
    "ldr s6,[%[b_ptr]],#4; ins v0.d[1],v2.d[0]\n\t"
    "sub %w[k_left],%w[k_left],#1\n\t"
    "fmla %[q1].4s,v0.4s,v4.s[0]\n\t"
    "fmla %[q2].4s,v0.4s,v5.s[0]\n\t"
    "fmla %[q3].4s,v0.4s,v6.s[0]\n\t"
    "6:\n\t"
   :[q1]"=w"(cq1), [q2]"=w"(cq2), [q3]"=w"(cq3), [k_left]"+r"(k_left),
    [a_ptr1]"+r"(a_ptr1), [a_ptr2]"+r"(a_ptr2), [a_ptr3]"+r"(a_ptr3),
    [a_ptr4]"+r"(a_ptr4), [b_ptr]"+r"(b_ptr), [next_pref]"+r"(next_pref)
   ::"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
   "v10","v11","v12","v13","v14","v15","v16","v17","v18");

  if (c_rowmajor) {
    float32x4x3_t cqt1 = vld3q_f32(c_ptr);
    cqt1.val[0] = vfmaq_n_f32(cq1, cqt1.val[0], beta);
    cqt1.val[1] = vfmaq_n_f32(cq2, cqt1.val[1], beta);
    cqt1.val[2] = vfmaq_n_f32(cq3, cqt1.val[2], beta);
    vst3q_f32(c_ptr, cqt1);
  } else {
    cq1 = vfmaq_n_f32(cq1, vld1q_f32(c_ptr), beta);
    cq2 = vfmaq_n_f32(cq2, vld1q_f32(c_ptr + LDM), beta);
    cq3 = vfmaq_n_f32(cq3, vld1q_f32(c_ptr + LDM * 2), beta);

    vst1q_f32(c_ptr, cq1); c_ptr += LDM;
    vst1q_f32(c_ptr, cq2); c_ptr += LDM;
    vst1q_f32(c_ptr, cq3);
  }
}

static inline void inline_sgemm_arowmajor_bskinny_m1n3(const float *a_ptr,
  const float *b_scr, float *c_ptr, uint32_t k_inc, uint32_t LDK,
  uint32_t LDM, float beta, bool c_rowmajor) {

  const float *sb_ptr = b_scr;

  float32x4_t cq01, cq02, cq03, cq04, cq05, cq06;
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = vdupq_n_f32(0.0f);
  float32x4_t cq07, cq08, cq09, cq10, cq11, cq12;
  cq07 = cq08 = cq09 = cq10 = cq11 = cq12 = vdupq_n_f32(0.0f);
  float32x4_t aq1, aq2, bq01, bq02, bq03, bq04, bq05, bq06;
  float32x4_t aq3, aq4, bq07, bq08, bq09, bq10, bq11, bq12;
  uint32_t k_left = k_inc;
  if (k_left > 7) {
    aq1 = vld1q_f32(a_ptr); aq2 = vld1q_f32(a_ptr + 4); a_ptr += 8;
    bq01 = vld1q_f32(sb_ptr); bq02 = vld1q_f32(sb_ptr + 4);
    bq03 = vld1q_f32(sb_ptr + 8); bq04 = vld1q_f32(sb_ptr + 12);
    bq05 = vld1q_f32(sb_ptr + 16); bq06 = vld1q_f32(sb_ptr + 20);
    sb_ptr += 24;
  }
  for (; k_left > 23; k_left -= 16) {
    aq3 = vld1q_f32(a_ptr);
    cq01 = vfmaq_f32(cq01, aq1, bq01); bq07 = vld1q_f32(sb_ptr);
    cq02 = vfmaq_f32(cq02, aq1, bq02); bq08 = vld1q_f32(sb_ptr + 4);
    cq03 = vfmaq_f32(cq03, aq1, bq03); bq09 = vld1q_f32(sb_ptr + 8);
    aq4 = vld1q_f32(a_ptr + 4);
    cq04 = vfmaq_f32(cq04, aq2, bq04); bq10 = vld1q_f32(sb_ptr + 12);
    cq05 = vfmaq_f32(cq05, aq2, bq05); bq11 = vld1q_f32(sb_ptr + 16);
    cq06 = vfmaq_f32(cq06, aq2, bq06); bq12 = vld1q_f32(sb_ptr + 20);
    aq1 = vld1q_f32(a_ptr + 8);
    cq07 = vfmaq_f32(cq07, aq3, bq07); bq01 = vld1q_f32(sb_ptr + 24);
    cq08 = vfmaq_f32(cq08, aq3, bq08); bq02 = vld1q_f32(sb_ptr + 28);
    cq09 = vfmaq_f32(cq09, aq3, bq09); bq03 = vld1q_f32(sb_ptr + 32);
    aq2 = vld1q_f32(a_ptr + 12); a_ptr += 16;
    cq10 = vfmaq_f32(cq10, aq4, bq10); bq04 = vld1q_f32(sb_ptr + 36);
    cq11 = vfmaq_f32(cq11, aq4, bq11); bq05 = vld1q_f32(sb_ptr + 40);
    cq12 = vfmaq_f32(cq12, aq4, bq12); bq06 = vld1q_f32(sb_ptr + 44);
    sb_ptr += 48;
  }
  if (k_left > 15) {
    aq3 = vld1q_f32(a_ptr);
    cq01 = vfmaq_f32(cq01, aq1, bq01); bq07 = vld1q_f32(sb_ptr);
    cq02 = vfmaq_f32(cq02, aq1, bq02); bq08 = vld1q_f32(sb_ptr + 4);
    cq03 = vfmaq_f32(cq03, aq1, bq03); bq09 = vld1q_f32(sb_ptr + 8);
    aq4 = vld1q_f32(a_ptr + 4); a_ptr += 8;
    cq04 = vfmaq_f32(cq04, aq2, bq04); bq10 = vld1q_f32(sb_ptr + 12);
    cq05 = vfmaq_f32(cq05, aq2, bq05); bq11 = vld1q_f32(sb_ptr + 16);
    cq06 = vfmaq_f32(cq06, aq2, bq06); bq12 = vld1q_f32(sb_ptr + 20);
    cq07 = vfmaq_f32(cq07, aq3, bq07); sb_ptr += 24;
    cq08 = vfmaq_f32(cq08, aq3, bq08); k_left -= 16;
    cq09 = vfmaq_f32(cq09, aq3, bq09);
    cq10 = vfmaq_f32(cq10, aq4, bq10);
    cq11 = vfmaq_f32(cq11, aq4, bq11);
    cq12 = vfmaq_f32(cq12, aq4, bq12);
  }
  if (k_left > 7) {
    cq01 = vfmaq_f32(cq01, aq1, bq01); k_left -= 8;
    cq02 = vfmaq_f32(cq02, aq1, bq02);
    cq03 = vfmaq_f32(cq03, aq1, bq03);
    cq04 = vfmaq_f32(cq04, aq2, bq04);
    cq05 = vfmaq_f32(cq05, aq2, bq05);
    cq06 = vfmaq_f32(cq06, aq2, bq06);
  }
  cq01 = vaddq_f32(cq01, cq07); cq02 = vaddq_f32(cq02, cq08);
  cq03 = vaddq_f32(cq03, cq09); cq04 = vaddq_f32(cq04, cq10);
  cq05 = vaddq_f32(cq05, cq11); cq06 = vaddq_f32(cq06, cq12);
  cq01 = vaddq_f32(cq01, cq04); cq02 = vaddq_f32(cq02, cq05);
  cq03 = vaddq_f32(cq03, cq06);

  if (k_left > 3) {
    aq1 = vld1q_f32(a_ptr); a_ptr += 4;
    bq01 = vld1q_f32(sb_ptr); bq02 = vld1q_f32(sb_ptr + 4);
    bq03 = vld1q_f32(sb_ptr + 8); sb_ptr += 12;
    cq01 = vfmaq_f32(cq01, aq1, bq01); k_left -= 4;
    cq02 = vfmaq_f32(cq02, aq1, bq02);
    cq03 = vfmaq_f32(cq03, aq1, bq03);
  }
  float32x2_t cd1 = vadd_f32(vget_low_f32(cq01), vget_high_f32(cq01));
  float32x2_t cd2 = vadd_f32(vget_low_f32(cq02), vget_high_f32(cq02));
  float32x2_t cd3 = vadd_f32(vget_low_f32(cq03), vget_high_f32(cq03));
  if (k_left > 1) {
    float32x2_t ad1 = vld1_f32(a_ptr); a_ptr += 2;
    float32x2_t bd1 = vld1_f32(sb_ptr);
    float32x2_t bd2 = vld1_f32(sb_ptr + 2);
    float32x2_t bd3 = vld1_f32(sb_ptr + 4); sb_ptr += 6;
    cd1 = vfma_f32(cd1, ad1, bd1); k_left -= 2;
    cd2 = vfma_f32(cd2, ad1, bd2);
    cd3 = vfma_f32(cd3, ad1, bd3);
  }
  float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);
  float cs2 = vget_lane_f32(cd2, 0) + vget_lane_f32(cd2, 1);
  float cs3 = vget_lane_f32(cd3, 0) + vget_lane_f32(cd3, 1);
  if (k_left > 0) {
    float as1 = *a_ptr++;
    cs1 += as1 * sb_ptr[0];
    cs2 += as1 * sb_ptr[1];
    cs3 += as1 * sb_ptr[2];
  }

  if (c_rowmajor) {
    c_ptr[0] = c_ptr[0] * beta + cs1;
    c_ptr[1] = c_ptr[1] * beta + cs2;
    c_ptr[2] = c_ptr[2] * beta + cs3;
  } else {
    c_ptr[0] = c_ptr[0] * beta + cs1;
    c_ptr[LDM] = c_ptr[LDM] * beta + cs2;
    c_ptr[LDM * 2] = c_ptr[LDM * 2] * beta + cs3;
  }
}

#define DEFAULT_SGEMV1_THRESH_K_UNROLL_M 512
#define DEFAULT_SGEMV1_THRESH_DETECT_CPU 30000

static inline bool unroll_test_m4n1(uint32_t M, uint32_t K) {
  unsigned char cpu_type = 0, cpu_id = 0;
  uint32_t gemv1_thresh_k_unroll_m = DEFAULT_SGEMV1_THRESH_K_UNROLL_M;
  if ((uint64_t)M * (uint64_t)K > DEFAULT_SGEMV1_THRESH_DETECT_CPU) {
    cpu_id = sched_getcpu();
    cpu_type = blas_arm_get_cpu_type(cpu_id);
    /* Based on a number of BLAS tests,
     * unrolling M on Cortex-A55 degrades performance in all cases */
    /* Unrolling M on other ARM cores can improve performance when K is small */
    gemv1_thresh_k_unroll_m = cpu_type == 55 ?
      0 : DEFAULT_SGEMV1_THRESH_K_UNROLL_M;
  }
  return K <= gemv1_thresh_k_unroll_m;
}

static inline bool unroll_test_m1n1(uint32_t M, uint32_t K) {
  return true;
}

static inline bool unroll_test_m4n2(uint32_t M, uint32_t K) {
  return unroll_test_m4n1(M, K);
}

static inline bool unroll_test_m1n2(uint32_t M, uint32_t K) {
  return true;
}

static inline bool unroll_test_m4n3(uint32_t M, uint32_t K) {
  unsigned char cpu_type = 0, cpu_id = 0;
  if ((uint64_t)M * (uint64_t)K > DEFAULT_SGEMV1_THRESH_DETECT_CPU) {
    cpu_id = sched_getcpu();
    cpu_type = blas_arm_get_cpu_type(cpu_id);
    if (cpu_type == 53 || cpu_type == 35) {
      return true;
    }
    return false;
  }
  return false;
}

static inline bool unroll_test_m1n3(uint32_t M, uint32_t K) {
  return true;
}

GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(sgemm, 1, 7, 5, 32768, float, float, unroll_test)
GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(sgemm, 2, 7, 5, 32768, float, float, unroll_test)
GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(sgemm, 3, 7, 5, 32768, float, float, unroll_test)

#define SGEMM_SKINNY1_FUNC_TEMPLATE(ndim) \
void sgemm_arowmajor_bskinny_afloat_bfloat_n##ndim(\
  const float *A, const float *B, float *C,\
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta) {\
\
  unsigned char cpu_type = 0;\
  if ((uint64_t)M * (uint64_t)K * ndim > \
    DEFAULT_SGEMV1_THRESH_DETECT_CPU << 3) {\
    unsigned char cpu_id = sched_getcpu();\
    cpu_type = blas_arm_get_cpu_type(cpu_id);\
  }\
\
  const uint32_t LDB = (b_c_order & 1) ? ndim : K;\
  const uint32_t LDC = (b_c_order & 2) ? ndim : M;\
  if (cpu_type == 35) {\
    sgemm_skinny1_arowmajor_n##ndim##_a35(A, B, C, M, K, K, LDB, LDC,\
      b_c_order, beta);\
  } else if (cpu_type == 53 || cpu_type == 55) {\
    sgemm_skinny1_arowmajor_n##ndim##_a53(A, B, C, M, K, K, LDB, LDC,\
      b_c_order, beta);\
  } else {\
    sgemm_skinny1_arowmajor_n##ndim##_a7x(A, B, C, M, K, K, LDB, LDC,\
      b_c_order, beta);\
  }\
}\
\
void sgemm_arowmajor_bskinny_afloat_bfloat_n##ndim##_omp(const float *A,\
  const float *B, float *C,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  float beta, uint32_t num_threads) {\
\
  const uint32_t LDC = (b_c_order & 2) ? ndim : M;\
  if (num_threads <= 1) {\
    sgemm_arowmajor_bskinny_afloat_bfloat_n##ndim(A, B, C, M, K,\
      b_c_order, beta);\
    return;\
  }\
\
  unsigned char cpu_type = 0;\
  if ((uint64_t)M * (uint64_t)K * ndim > \
    DEFAULT_SGEMV1_THRESH_DETECT_CPU << 3) {\
    unsigned char cpu_id = sched_getcpu();\
    cpu_type = blas_arm_get_cpu_type(cpu_id);\
  }\
\
  const uint32_t LDB = (b_c_order & 1) ? ndim : K;\
  if (cpu_type == 35) {\
    sgemm_skinny1_arowmajor_n##ndim##_a35_omp(A, B, C, M, K, K,\
      LDB, LDC, b_c_order, beta, num_threads);\
  } else if (cpu_type == 53 || cpu_type == 55) {\
    sgemm_skinny1_arowmajor_n##ndim##_a53_omp(A, B, C, M, K, K,\
      LDB, LDC, b_c_order, beta, num_threads);\
  } else {\
    sgemm_skinny1_arowmajor_n##ndim##_a7x_omp(A, B, C, M, K, K,\
      LDB, LDC, b_c_order, beta, num_threads);\
  }\
}

SGEMM_SKINNY1_FUNC_TEMPLATE(4)
SGEMM_SKINNY1_FUNC_TEMPLATE(5)
SGEMM_SKINNY1_FUNC_TEMPLATE(6)
SGEMM_SKINNY1_FUNC_TEMPLATE(7)
SGEMM_SKINNY1_FUNC_TEMPLATE(8)
SGEMM_SKINNY1_FUNC_TEMPLATE(9)
SGEMM_SKINNY1_FUNC_TEMPLATE(10)
SGEMM_SKINNY1_FUNC_TEMPLATE(11)
SGEMM_SKINNY1_FUNC_TEMPLATE(12)
SGEMM_SKINNY1_FUNC_TEMPLATE(13)
SGEMM_SKINNY1_FUNC_TEMPLATE(14)
SGEMM_SKINNY1_FUNC_TEMPLATE(15)
SGEMM_SKINNY1_FUNC_TEMPLATE(16)
SGEMM_SKINNY1_FUNC_TEMPLATE(17)
SGEMM_SKINNY1_FUNC_TEMPLATE(18)
SGEMM_SKINNY1_FUNC_TEMPLATE(19)
SGEMM_SKINNY1_FUNC_TEMPLATE(20)
SGEMM_SKINNY1_FUNC_TEMPLATE(21)
SGEMM_SKINNY1_FUNC_TEMPLATE(22)
SGEMM_SKINNY1_FUNC_TEMPLATE(23)
SGEMM_SKINNY1_FUNC_TEMPLATE(24)
SGEMM_SKINNY1_FUNC_TEMPLATE(25)
SGEMM_SKINNY1_FUNC_TEMPLATE(26)
SGEMM_SKINNY1_FUNC_TEMPLATE(27)
SGEMM_SKINNY1_FUNC_TEMPLATE(28)
SGEMM_SKINNY1_FUNC_TEMPLATE(29)
SGEMM_SKINNY1_FUNC_TEMPLATE(30)
SGEMM_SKINNY1_FUNC_TEMPLATE(31)
SGEMM_SKINNY1_FUNC_TEMPLATE(32)
SGEMM_SKINNY1_FUNC_TEMPLATE(33)
SGEMM_SKINNY1_FUNC_TEMPLATE(34)
SGEMM_SKINNY1_FUNC_TEMPLATE(35)
SGEMM_SKINNY1_FUNC_TEMPLATE(36)
SGEMM_SKINNY1_FUNC_TEMPLATE(37)
SGEMM_SKINNY1_FUNC_TEMPLATE(38)
SGEMM_SKINNY1_FUNC_TEMPLATE(39)
SGEMM_SKINNY1_FUNC_TEMPLATE(40)
SGEMM_SKINNY1_FUNC_TEMPLATE(41)
SGEMM_SKINNY1_FUNC_TEMPLATE(42)
SGEMM_SKINNY1_FUNC_TEMPLATE(43)
SGEMM_SKINNY1_FUNC_TEMPLATE(44)
SGEMM_SKINNY1_FUNC_TEMPLATE(45)
SGEMM_SKINNY1_FUNC_TEMPLATE(46)
SGEMM_SKINNY1_FUNC_TEMPLATE(47)
SGEMM_SKINNY1_FUNC_TEMPLATE(48)
SGEMM_SKINNY1_FUNC_TEMPLATE(49)
SGEMM_SKINNY1_FUNC_TEMPLATE(50)

