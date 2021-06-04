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


#include "arm_neon/ARMCpuType.h"
#include "arm_neon/ARMCompareAndSwap.h"
#include "common/CommonSkinnyDot.h"
#include <arm_neon.h>

typedef float sgemm_skinnydot_ascalar;
typedef float sgemm_skinnydot_bscalar;
typedef float sgemm_skinnydot_cscalar;

static inline void inline_sgemm_arowmajor_bskinny_m4n1(const float *a_ptr1,
  const float *b_ptr, float *c_ptr, uint32_t k_inc, uint32_t LDK, uint32_t LDM,
  float beta, bool c_rowmajor) {

  float32x2_t cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8;
  const float *a_ptr2 = a_ptr1 + LDK;
  const float *a_ptr3 = a_ptr1 + LDK * 2;
  const float *a_ptr4 = a_ptr2 + LDK * 2;
  const float *a_pref = a_ptr4 + LDK;
  const uint32_t pref_inc = LDK > k_inc ? (LDK - k_inc) * sizeof(float) : 0;
  uint32_t k_left = k_inc;
  __asm__ __volatile__(
    "mov r0,#0\n\t"
    "vmov.i8 %[cd1],#0; vmov.i8 %[cd2],#0\n\t"
    "vmov.i8 %[cd3],#0; vmov.i8 %[cd4],#0\n\t"
    "vmov.i8 %[cd5],#0; vmov.i8 %[cd6],#0\n\t"
    "vmov.i8 %[cd7],#0; vmov.i8 %[cd8],#0\n\t"
    "cmp %[k_left],#4; blt 3f\n\t"
    "vldr d2,[%[a_ptr1]]; vldr d6,[%[a_ptr1],#8]; add %[a_ptr1],%[a_ptr1],#16\n\t"
    "vldr d3,[%[a_ptr2]]; vldr d7,[%[a_ptr2],#8]; add %[a_ptr2],%[a_ptr2],#16\n\t"
    "vldr d4,[%[a_ptr3]]; vldr d8,[%[a_ptr3],#8]; add %[a_ptr3],%[a_ptr3],#16\n\t"
    "vldr d5,[%[a_ptr4]]; vldr d9,[%[a_ptr4],#8]; add %[a_ptr4],%[a_ptr4],#16\n\t"
    "vldm %[b_ptr]!,{d0,d1}\n\t"
    "cmp %[k_left],#8; blt 2f\n\t"
    ".balign 16; 1:\n\t"
    "pld [%[a_pref]]; add %[a_pref],%[a_pref],#64; add r0,r0,#16\n\t"
    "vmla.f32 %[cd1],d2,d0; vldr d2,[%[a_ptr1]]\n\t"
    "cmp r0,%[k_inc]\n\t"
    "vmla.f32 %[cd2],d3,d0; vldr d3,[%[a_ptr2]]\n\t"
    "addgt %[a_pref],%[a_pref],%[pref_inc]\n\t"
    "vmla.f32 %[cd3],d4,d0; vldr d4,[%[a_ptr3]]\n\t"
    "movgt r0,#0\n\t"
    "vmla.f32 %[cd4],d5,d0; vldr d5,[%[a_ptr4]]\n\t"
    "vldr d0,[%[b_ptr]]; sub %[k_left],%[k_left],#4\n\t"
    "vmla.f32 %[cd5],d6,d1; vldr d6,[%[a_ptr1],#8]\n\t"
    "add %[a_ptr1],%[a_ptr1],#16\n\t"
    "vmla.f32 %[cd6],d7,d1; vldr d7,[%[a_ptr2],#8]\n\t"
    "add %[a_ptr2],%[a_ptr2],#16; cmp %[k_left],#8\n\t"
    "vmla.f32 %[cd7],d8,d1; vldr d8,[%[a_ptr3],#8]\n\t"
    "add %[a_ptr3],%[a_ptr3],#16\n\t"
    "vmla.f32 %[cd8],d9,d1; vldr d9,[%[a_ptr4],#8]\n\t"
    "add %[a_ptr4],%[a_ptr4],#16\n\t"
    "vldr d1,[%[b_ptr],#8]; add %[b_ptr],%[b_ptr],#16; bge 1b\n\t"
    "2:\n\t"
    "vmla.f32 %[cd1],d2,d0; vmla.f32 %[cd2],d3,d0\n\t"
    "vmla.f32 %[cd3],d4,d0; vmla.f32 %[cd4],d5,d0\n\t"
    "sub %[k_left],%[k_left],#4\n\t"
    "vmla.f32 %[cd5],d6,d1; vmla.f32 %[cd6],d7,d1\n\t"
    "vmla.f32 %[cd7],d8,d1; vmla.f32 %[cd8],d9,d1\n\t"
    "3:\n\t"
   :[cd1]"=w"(cd1), [cd2]"=w"(cd2), [cd3]"=w"(cd3), [cd4]"=w"(cd4),
    [cd5]"=w"(cd5), [cd6]"=w"(cd6), [cd7]"=w"(cd7), [cd8]"=w"(cd8),
    [a_ptr1]"+r"(a_ptr1), [a_ptr2]"+r"(a_ptr2), [a_ptr3]"+r"(a_ptr3),
    [a_ptr4]"+r"(a_ptr4), [b_ptr]"+r"(b_ptr),
    [k_left]"+r"(k_left), [a_pref]"+r"(a_pref)
   :[pref_inc]"r"(pref_inc), [k_inc]"r"(k_inc)
   :"d0","d1","d2","d3","d4","d5","d6","d7","d8","d9",
    "r0","cc","memory");

  cd1 = vadd_f32(cd1, cd5); cd2 = vadd_f32(cd2, cd6);
  cd3 = vadd_f32(cd3, cd7); cd4 = vadd_f32(cd4, cd8);
  float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);
  float cs2 = vget_lane_f32(cd2, 0) + vget_lane_f32(cd2, 1);
  float cs3 = vget_lane_f32(cd3, 0) + vget_lane_f32(cd3, 1);
  float cs4 = vget_lane_f32(cd4, 0) + vget_lane_f32(cd4, 1);
  for (; k_left > 0; k_left--) {
    float bs1 = *b_ptr; b_ptr++;
    cs1 += (*a_ptr1) * bs1; a_ptr1++; 
    cs2 += (*a_ptr2) * bs1; a_ptr2++;
    cs3 += (*a_ptr3) * bs1; a_ptr3++; 
    cs4 += (*a_ptr4) * bs1; a_ptr4++;
  }
  c_ptr[0] = c_ptr[0] * beta + cs1; c_ptr[1] = c_ptr[1] * beta + cs2;
  c_ptr[2] = c_ptr[2] * beta + cs3; c_ptr[3] = c_ptr[3] * beta + cs4;
}

static inline void inline_sgemm_arowmajor_bskinny_m1n1(const float *a_ptr,
  const float *b_ptr, float *c_ptr, uint32_t k_left, uint32_t LDK, uint32_t LDM,
  float beta, bool c_rowmajor) {

  float32x4_t cq1;
  __asm__ __volatile__(
    "vmov.i8 d16,#0; vmov.i8 d17,#0\n\t"
    "vmov d18,d16; vmov d19,d17\n\t"
    "vmov d20,d16; vmov d21,d17\n\t"
    "vmov d22,d16; vmov d23,d17\n\t"
    "cmp %[K],#16; blt 4f\n\t"
    "pld [%[a_ptr],#256]\n\t"
    "add %[a_ptr],%[a_ptr],#64; add %[b_ptr],%[b_ptr],#64\n\t"
    "vldr d24,[%[a_ptr],#-64]; vldr d8,[%[b_ptr],#-64]\n\t"
    "vldr d25,[%[a_ptr],#-56]; vldr d9,[%[b_ptr],#-56]\n\t"
    "vldr d26,[%[a_ptr],#-48]; vldr d10,[%[b_ptr],#-48]\n\t"
    "vldr d27,[%[a_ptr],#-40]; vldr d11,[%[b_ptr],#-40]\n\t"
    "vldr d28,[%[a_ptr],#-32]; vldr d12,[%[b_ptr],#-32]\n\t"
    "vldr d29,[%[a_ptr],#-24]; vldr d13,[%[b_ptr],#-24]\n\t"
    "vldr d30,[%[a_ptr],#-16]; vldr d14,[%[b_ptr],#-16]\n\t"
    "vldr d31,[%[a_ptr],#-8]; vldr d15,[%[b_ptr],#-8]\n\t"
    "cmp %[K],#32; blt 3f\n\t"
    "2:\n\t"
    "pld [%[a_ptr],#256]\n\t"
    "add %[a_ptr],%[a_ptr],#64; add %[b_ptr],%[b_ptr],#64\n\t"
    "vmla.f32 d16,d24,d8; vldr d24,[%[a_ptr],#-64]; vldr d8,[%[b_ptr],#-64]\n\t"
    "vmla.f32 d17,d25,d9; vldr d25,[%[a_ptr],#-56]; vldr d9,[%[b_ptr],#-56]\n\t"
    "vmla.f32 d18,d26,d10; vldr d26,[%[a_ptr],#-48]; vldr d10,[%[b_ptr],#-48]\n\t"
    "vmla.f32 d19,d27,d11; vldr d27,[%[a_ptr],#-40]; vldr d11,[%[b_ptr],#-40]\n\t"
    "sub %[K],%[K],#16\n\t"
    "vmla.f32 d20,d28,d12; vldr d28,[%[a_ptr],#-32]; vldr d12,[%[b_ptr],#-32]\n\t"
    "vmla.f32 d21,d29,d13; vldr d29,[%[a_ptr],#-24]; vldr d13,[%[b_ptr],#-24]\n\t"
    "cmp %[K],#32\n\t"
    "vmla.f32 d22,d30,d14; vldr d30,[%[a_ptr],#-16]; vldr d14,[%[b_ptr],#-16]\n\t"
    "vmla.f32 d23,d31,d15; vldr d31,[%[a_ptr],#-8]; vldr d15,[%[b_ptr],#-8]\n\t"
    "bge 2b\n\t"
    "3:\n\t"
    "vmla.f32 d16,d24,d8; vmla.f32 d17,d25,d9\n\t"
    "vmla.f32 d18,d26,d10; vmla.f32 d19,d27,d11; sub %[K],%[K],#16\n\t"
    "vmla.f32 d20,d28,d12; vmla.f32 d21,d29,d13\n\t"
    "vmla.f32 d22,d30,d14; vmla.f32 d23,d31,d15\n\t"
    "4:\n\t"
    "vadd.f32 d16,d16,d20; vadd.f32 d17,d17,d21\n\t"
    "vadd.f32 d18,d18,d22; vadd.f32 d19,d19,d23\n\t"
    "cmp %[K],#8; blt 5f; add %[a_ptr],%[a_ptr],#32; add %[b_ptr],%[b_ptr],#32\n\t"
    "vldr d24,[%[a_ptr],#-32]; vldr d8,[%[b_ptr],#-32]; vmla.f32 d16,d24,d8\n\t"
    "vldr d25,[%[a_ptr],#-24]; vldr d9,[%[b_ptr],#-24]; vmla.f32 d17,d25,d9\n\t"
    "sub %[K],%[K],#8\n\t"
    "vldr d26,[%[a_ptr],#-16]; vldr d10,[%[b_ptr],#-16]; vmla.f32 d18,d26,d10\n\t"
    "vldr d27,[%[a_ptr],#-8]; vldr d11,[%[b_ptr],#-8]; vmla.f32 d19,d27,d11\n\t"
    "5:\n\t"
    "vadd.f32 %e[cq1],d16,d17; vadd.f32 %f[cq1],d18,d19\n\t"
    "cmp %[K],#4; blt 6f\n\t"
    "add %[a_ptr],%[a_ptr],#16; add %[b_ptr],%[b_ptr],#16\n\t"
    "vldr d24,[%[a_ptr],#-16]; vldr d8,[%[b_ptr],#-16]; vmla.f32 %e[cq1],d24,d8\n\t"
    "sub %[K],%[K],#4\n\t"
    "vldr d25,[%[a_ptr],#-8]; vldr d9,[%[b_ptr],#-8]; vmla.f32 %f[cq1],d25,d9\n\t"
    "6:\n\t"
  :[cq1]"=w"(cq1), [a_ptr]"+r"(a_ptr), [b_ptr]"+r"(b_ptr), [K]"+r"(k_left)
  ::"cc","memory","q12","q13","q14","q15",
    "q4","q5","q6","q7","q8","q9","q10","q11");

  float32x2_t cd1 = vadd_f32(vget_low_f32(cq1), vget_high_f32(cq1));
  if (k_left > 1) {
    float32x2_t ad1 = vld1_f32(a_ptr); a_ptr += 2;
    float32x2_t bd1 = vld1_f32(b_ptr); b_ptr += 2;
    cd1 = vmla_f32(cd1, ad1, bd1);
    k_left -= 2;
  }

  float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);
  if (k_left > 0) {
    cs1 += a_ptr[0] * b_ptr[0];
  }
  c_ptr[0] = c_ptr[0] * beta + cs1;
}

/* k_mask = 7 */
static inline void inline_sgemm_arowmajor_bskinny_m4n2(const float *a_ptr1,
  const float *b_ptr, float *c_ptr, uint32_t k_inc, uint32_t LDK, uint32_t LDM,
  float beta, bool c_rowmajor) {

  const float *a_ptr2 = a_ptr1 + LDK;
  const float *a_ptr3 = a_ptr1 + LDK * 2;
  const float *a_ptr4 = a_ptr1 + LDK * 3;
  const float *a_pref = a_ptr1 + LDK * 4;
  uint32_t k_left = k_inc;
  const uint32_t pref_inc = LDK > k_inc ? (LDK - k_inc) * sizeof(float) : 0;
  float32x4_t cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8;
  __asm__ __volatile__(
    "mov r0,#0\n\t"
    "vmov.i8 %q[cq1],#0; vmov.i8 %q[cq2],#0\n\t"
    "vmov.i8 %q[cq3],#0; vmov.i8 %q[cq4],#0\n\t"
    "vmov.i8 %q[cq5],#0; vmov.i8 %q[cq6],#0\n\t"
    "vmov.i8 %q[cq7],#0; vmov.i8 %q[cq8],#0\n\t"
    "cmp %[k_left],#4; blt 3f\n\t"
    "vldm %[a_ptr1]!,{q2}; vldm %[a_ptr2]!,{q3}\n\t"
    "vldm %[a_ptr3]!,{q4}; vldm %[a_ptr4]!,{q5}\n\t"
    "vldm %[b_ptr]!,{q0}; vldm %[b_ptr]!,{q1}\n\t"
    "cmp %[k_left],#8; blt 2f\n\t"
    ".balign 16; 1:\n\t"
    "pld [%[a_pref]]; add %[a_pref],%[a_pref],#64; add r0,r0,#16\n\t"
    "vmla.f32 %q[cq1],q2,q0; cmp r0,%[k_inc]\n\t"
    "vmla.f32 %q[cq5],q2,q1; vldm %[a_ptr1]!,{q2}\n\t"
    "vmla.f32 %q[cq2],q3,q0; addgt %[a_pref],%[a_pref],%[pref_inc]\n\t"
    "vmla.f32 %q[cq6],q3,q1; vldm %[a_ptr2]!,{q3}\n\t"
    "sub %[k_left],%[k_left],#4\n\t"
    "vmla.f32 %q[cq3],q4,q0; movgt r0,#0\n\t"
    "vmla.f32 %q[cq7],q4,q1; vldm %[a_ptr3]!,{q4}\n\t"
    "vmla.f32 %q[cq4],q5,q0; cmp %[k_left],#8\n\t"
    "vmla.f32 %q[cq8],q5,q1; vldm %[a_ptr4]!,{q5}\n\t"
    "vldm %[b_ptr]!,{q0}; vldm %[b_ptr]!,{q1}; bge 1b\n\t"
    "2:\n\t"
    "vmla.f32 %q[cq1],q2,q0; vmla.f32 %q[cq5],q2,q1\n\t"
    "vmla.f32 %q[cq2],q3,q0; vmla.f32 %q[cq6],q3,q1\n\t"
    "vmla.f32 %q[cq3],q4,q0; vmla.f32 %q[cq7],q4,q1\n\t"
    "vmla.f32 %q[cq4],q5,q0; vmla.f32 %q[cq8],q5,q1\n\t"
    "sub %[k_left],%[k_left],#4\n\t"
    "3:\n\t"
   :[cq1]"=w"(cq1), [cq2]"=w"(cq2), [cq3]"=w"(cq3), [cq4]"=w"(cq4),
    [cq5]"=w"(cq5), [cq6]"=w"(cq6), [cq7]"=w"(cq7), [cq8]"=w"(cq8),
    [k_left]"+r"(k_left), [a_pref]"+r"(a_pref), [b_ptr]"+r"(b_ptr),
    [a_ptr1]"+r"(a_ptr1), [a_ptr2]"+r"(a_ptr2),
    [a_ptr3]"+r"(a_ptr3), [a_ptr4]"+r"(a_ptr4)
   :[pref_inc]"r"(pref_inc), [k_inc]"r"(k_inc)
   :"d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11",
    "r0","cc","memory");

  float32x2_t cd1 = vadd_f32(vget_low_f32(cq1), vget_high_f32(cq1));
  float32x2_t cd2 = vadd_f32(vget_low_f32(cq2), vget_high_f32(cq2));
  float32x2_t cd3 = vadd_f32(vget_low_f32(cq3), vget_high_f32(cq3));
  float32x2_t cd4 = vadd_f32(vget_low_f32(cq4), vget_high_f32(cq4));
  float32x2_t cd5 = vadd_f32(vget_low_f32(cq5), vget_high_f32(cq5));
  float32x2_t cd6 = vadd_f32(vget_low_f32(cq6), vget_high_f32(cq6));
  float32x2_t cd7 = vadd_f32(vget_low_f32(cq7), vget_high_f32(cq7));
  float32x2_t cd8 = vadd_f32(vget_low_f32(cq8), vget_high_f32(cq8));
  if (k_left >= 2) {
    float32x2_t bd1 = vld1_f32(b_ptr);
    float32x2_t bd2 = vld1_f32(b_ptr + 2); b_ptr += 4;
    float32x2_t ad1 = vld1_f32(a_ptr1); a_ptr1 += 2;
    float32x2_t ad2 = vld1_f32(a_ptr2); a_ptr2 += 2;
    float32x2_t ad3 = vld1_f32(a_ptr3); a_ptr3 += 2;
    float32x2_t ad4 = vld1_f32(a_ptr4); a_ptr4 += 2;
    cd1 = vmla_f32(cd1, ad1, bd1);
    cd2 = vmla_f32(cd2, ad2, bd1);
    cd3 = vmla_f32(cd3, ad3, bd1);
    cd4 = vmla_f32(cd4, ad4, bd1);
    cd5 = vmla_f32(cd5, ad1, bd2);
    cd6 = vmla_f32(cd6, ad2, bd2);
    cd7 = vmla_f32(cd7, ad3, bd2);
    cd8 = vmla_f32(cd8, ad4, bd2);
    k_left -= 2;
  }
  float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);
  float cs2 = vget_lane_f32(cd2, 0) + vget_lane_f32(cd2, 1);
  float cs3 = vget_lane_f32(cd3, 0) + vget_lane_f32(cd3, 1);
  float cs4 = vget_lane_f32(cd4, 0) + vget_lane_f32(cd4, 1);
  float cs5 = vget_lane_f32(cd5, 0) + vget_lane_f32(cd5, 1);
  float cs6 = vget_lane_f32(cd6, 0) + vget_lane_f32(cd6, 1);
  float cs7 = vget_lane_f32(cd7, 0) + vget_lane_f32(cd7, 1);
  float cs8 = vget_lane_f32(cd8, 0) + vget_lane_f32(cd8, 1);
  if (k_left > 0) {
    float bs1 = b_ptr[0];
    float bs2 = b_ptr[1];
    float as1 = *a_ptr1;
    float as2 = *a_ptr2;
    float as3 = *a_ptr3;
    float as4 = *a_ptr4;
    cs1 += as1 * bs1; cs2 += as2 * bs1;
    cs3 += as3 * bs1; cs4 += as4 * bs1;
    cs5 += as1 * bs2; cs6 += as2 * bs2;
    cs7 += as3 * bs2; cs8 += as4 * bs2;
  }
  if (c_rowmajor) {
    c_ptr[0] = c_ptr[0] * beta + cs1; c_ptr[1] = c_ptr[1] * beta + cs5;
    c_ptr[2] = c_ptr[2] * beta + cs2; c_ptr[3] = c_ptr[3] * beta + cs6;
    c_ptr[4] = c_ptr[4] * beta + cs3; c_ptr[5] = c_ptr[5] * beta + cs7;
    c_ptr[6] = c_ptr[6] * beta + cs4; c_ptr[7] = c_ptr[7] * beta + cs8;
  } else {
    c_ptr[0] = c_ptr[0] * beta + cs1; c_ptr[1] = c_ptr[1] * beta + cs2;
    c_ptr[2] = c_ptr[2] * beta + cs3; c_ptr[3] = c_ptr[3] * beta + cs4;
    c_ptr += LDM;
    c_ptr[0] = c_ptr[0] * beta + cs5; c_ptr[1] = c_ptr[1] * beta + cs6;
    c_ptr[2] = c_ptr[2] * beta + cs7; c_ptr[3] = c_ptr[3] * beta + cs8;
  }
}

static inline void inline_sgemm_arowmajor_bskinny_m1n2(const float *a_ptr,
  const float *b_ptr, float *c_ptr, uint32_t k_left, uint32_t LDK, uint32_t LDM,
  float beta, bool c_rowmajor) {

  register float32x4_t cq1 __asm("q8");
  __asm__ __volatile__(
    "vmov.i8 %q[cq1],#0; vmov.i8 q9,#0\n\t"
    "vmov.i8 q10,#0; vmov.i8 q11,#0\n\t"
    "cmp %[k_left],#16; blt 4f\n\t"
    "pld [%[a_ptr],#256]\n\t"
    "vldm %[a_ptr]!,{q12,q13,q14,q15}\n\t"
    "vldm %[b_ptr]!,{q0,q1,q2,q3}\n\t"
    "vldm %[b_ptr]!,{q4,q5,q6,q7}\n\t"
    "cmp %[k_left],#32; blt 3f\n\t"
    ".balign 16; 2:\n\t"
    "pld [%[a_ptr],#256]\n\t"
    "vmla.f32 %q[cq1],q12,q0; vldm %[b_ptr]!,{q0}\n\t"
    "vmla.f32 q10,q12,q1; vldm %[b_ptr]!,{q1}; vldm %[a_ptr]!,{q12}\n\t"
    "vmla.f32 q9,q13,q2; vldm %[b_ptr]!,{q2}\n\t"
    "vmla.f32 q11,q13,q3; vldm %[b_ptr]!,{q3}; vldm %[a_ptr]!,{q13}\n\t"
    "sub %[k_left],%[k_left],#16\n\t"
    "vmla.f32 %q[cq1],q14,q4; vldm %[b_ptr]!,{q4}\n\t"
    "vmla.f32 q10,q14,q5; vldm %[b_ptr]!,{q5}; vldm %[a_ptr]!,{q14}\n\t"
    "cmp %[k_left],#32\n\t"
    "vmla.f32 q9,q15,q6; vldm %[b_ptr]!,{q6}\n\t"
    "vmla.f32 q11,q15,q7; vldm %[b_ptr]!,{q7}; vldm %[a_ptr]!,{q15}\n\t"
    "bge 2b\n\t"
    "3:\n\t"
    "vmla.f32 %q[cq1],q12,q0; vmla.f32 q10,q12,q1; sub %[k_left],%[k_left],#16\n\t"
    "vmla.f32 q9,q13,q2; vmla.f32 q11,q13,q3\n\t"
    "vmla.f32 %q[cq1],q14,q4; vmla.f32 q10,q14,q5\n\t"
    "vmla.f32 q9,q15,q6; vmla.f32 q11,q15,q7\n\t"
    "4:\n\t"
    "cmp %[k_left],#8; blt 5f\n\t"
    "vldm %[a_ptr]!,{q12}; vldm %[b_ptr]!,{q0,q1}\n\t"
    "vldm %[a_ptr]!,{q13}; vldm %[b_ptr]!,{q2,q3}\n\t"
    "vmla.f32 %q[cq1],q12,q0; vmla.f32 q10,q12,q1\n\t"
    "sub %[k_left],%[k_left],#8\n\t"
    "vmla.f32 q9,q13,q2; vmla.f32 q11,q13,q3\n\t"
    "5:\n\t"
    "vadd.f32 %q[cq1],%q[cq1],q9; vadd.f32 q10,q10,q11\n\t"
    "cmp %[k_left],#4; blt 6f\n\t"
    "vldm %[a_ptr]!,{q12}; vldm %[b_ptr]!,{q4}; vldm %[b_ptr]!,{q0}\n\t"
    "vmla.f32 %q[cq1],q12,q4; vmla.f32 q10,q12,q0\n\t"
    "sub %[k_left],%[k_left],#4\n\t"
    "6:\n\t"
    "vadd.f32 %e[cq1],%e[cq1],%f[cq1]; vadd.f32 %f[cq1],d20,d21\n\t"
    "cmp %[k_left],#2; blt 7f\n\t"
    "vld1.32 {d24},[%[a_ptr]]!\n\t"
    "vld1.32 {d8},[%[b_ptr]]!; vld1.32 {d0},[%[b_ptr]]!\n\t"
    "vmla.f32 %e[cq1],d24,d8; vmla.f32 %f[cq1],d24,d0\n\t"
    "sub %[k_left],%[k_left],#2\n\t"
    "7:\n\t"
   :[cq1]"=w"(cq1), [a_ptr]"+r"(a_ptr),
    [k_left]"+r"(k_left), [b_ptr]"+r"(b_ptr)
   ::"cc","memory","q0","q1","q2","q3","q4","q5","q6","q7",
     "q9","q10","q11","q12","q13","q14","q15");

  float32x2_t cd1 = vpadd_f32(vget_low_f32(cq1), vget_high_f32(cq1));
  if (k_left > 0) {
    float as1 = *a_ptr;
    float32x2_t bd1 = vld1_f32(b_ptr);
    cd1 = vmla_n_f32(cd1, bd1, as1);
  }

  if (c_rowmajor) {
    cd1 = vmla_n_f32(cd1, vld1_f32(c_ptr), beta);
    vst1_f32(c_ptr, cd1);
  } else {
    c_ptr[0] = c_ptr[0] * beta + vget_lane_f32(cd1, 0);
    c_ptr[LDM] = c_ptr[LDM] * beta + vget_lane_f32(cd1, 1);
  }
}

static inline bool unroll_test_m4n1(uint32_t M, uint32_t K) {
  return K <= 512;
}

static inline bool unroll_test_m1n1(uint32_t M, uint32_t K) {
  return true;
}

static inline bool unroll_test_m4n2(uint32_t M, uint32_t K) {
  return K <= 512;
}

static inline bool unroll_test_m1n2(uint32_t M, uint32_t K) {
  return true;
}

GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(sgemm, 1, 5, 5, 32768, float, float, unroll_test)
GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(sgemm, 2, 7, 5, 32768, float, float, unroll_test)

typedef float sgemm_skinnydot_avec1;
typedef float sgemm_skinnydot_bvec1;
typedef float sgemm_skinnydot_cvec1;

typedef float32x2_t sgemm_skinnydot_avec2;
typedef float32x2_t sgemm_skinnydot_bvec2;
typedef float32x2_t sgemm_skinnydot_cvec2;

typedef float32x4_t sgemm_skinnydot_avec4;
typedef float32x4_t sgemm_skinnydot_bvec4;
typedef float32x4_t sgemm_skinnydot_cvec4;

typedef float32x4x2_t sgemm_skinnydot_avec8;
typedef float32x4x2_t sgemm_skinnydot_bvec8;
typedef float32x4x2_t sgemm_skinnydot_cvec8;

GEMM_SKINNY_DOT_CALC_UNIT(sgemm, 8) {
  float32x4x2_t ret;
  ret.val[0] = vmlaq_f32(c_vec.val[0], a_vec.val[0], b_vec.val[0]);
  ret.val[1] = vmlaq_f32(c_vec.val[1], a_vec.val[1], b_vec.val[1]);
  return ret;
}

GEMM_SKINNY_DOT_CALC_UNIT(sgemm, 4) {
  return vmlaq_f32(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_DOT_CALC_UNIT(sgemm, 2) {
  return vmla_f32(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_DOT_CALC_UNIT(sgemm, 1) {
  return c_vec + a_vec * b_vec;
}

GEMM_SKINNY_DOT_LOADA_UNIT(sgemm, 8) {
  __asm__("pld [%0,#96]"::"r"(a_ptr):);
  float32x4x2_t ret;
  ret.val[0] = vld1q_f32(a_ptr);
  ret.val[1] = vld1q_f32(a_ptr + 4);
  return ret;
}

GEMM_SKINNY_DOT_LOADA_UNIT(sgemm, 4) {
  __asm__("pld [%0,#80]"::"r"(a_ptr):);
  return vld1q_f32(a_ptr);
}

GEMM_SKINNY_DOT_LOADA_UNIT(sgemm, 2) {
  __asm__("pld [%0,#72]"::"r"(a_ptr):);
  return vld1_f32(a_ptr);
}

GEMM_SKINNY_DOT_LOADA_UNIT(sgemm, 1) {
  return *a_ptr;
}

GEMM_SKINNY_DOT_LOADB_UNIT(sgemm, 8) {
  float32x4x2_t ret;
  ret.val[0] = vld1q_f32(b_ptr);
  ret.val[1] = vld1q_f32(b_ptr + 4);
  return ret;
}

GEMM_SKINNY_DOT_LOADB_UNIT(sgemm, 4) {
  return vld1q_f32(b_ptr);
}

GEMM_SKINNY_DOT_LOADB_UNIT(sgemm, 2) {
  return vld1_f32(b_ptr);
}

GEMM_SKINNY_DOT_LOADB_UNIT(sgemm, 1) {
  return *b_ptr;
}

GEMM_SKINNY_DOT_REDUC_UNIT(sgemm, 8, 4) {
  return vaddq_f32(c_vec.val[0], c_vec.val[1]);
}

GEMM_SKINNY_DOT_REDUC_UNIT(sgemm, 4, 2) {
  return vadd_f32(vget_low_f32(c_vec), vget_high_f32(c_vec));
}

GEMM_SKINNY_DOT_REDUC_UNIT(sgemm, 2, 1) {
  return vget_lane_f32(c_vec, 0) + vget_lane_f32(c_vec, 1);
}

GEMM_SKINNY_DOT_INITC_UNIT(sgemm, 8) {
  float32x4x2_t ret;
  ret.val[0] = vdupq_n_f32(0);
  ret.val[1] = vdupq_n_f32(0);
  return ret;
}

GEMM_SKINNY_DOT_INITC_UNIT(sgemm, 4) {
  return vdupq_n_f32(0);
}

GEMM_SKINNY_DOT_INITC_UNIT(sgemm, 2) {
  return vdup_n_f32(0);
}

GEMM_SKINNY_DOT_INITC_UNIT(sgemm, 1) {
  return 0;
}

GEMM_SKINNY_DOT_PARALLEL_FUNC(sgemm, 3, 3, 7, 32768, float, float)
GEMM_SKINNY_DOT_PARALLEL_FUNC(sgemm, 4, 3, 7, 32768, float, float)
GEMM_SKINNY_DOT_PARALLEL_FUNC(sgemm, 5, 3, 7, 32768, float, float)
GEMM_SKINNY_DOT_PARALLEL_FUNC(sgemm, 6, 3, 7, 32768, float, float)
GEMM_SKINNY_DOT_PARALLEL_FUNC(sgemm, 7, 3, 3, 32768, float, float)
GEMM_SKINNY_DOT_PARALLEL_FUNC(sgemm, 8, 3, 3, 32768, float, float)
