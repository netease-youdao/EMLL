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


#include "arm_neon/ARMCompareAndSwap.h"
#include "common/CommonSkinnyDot.h"
#include <arm_neon.h>

static inline void inline_hgemm_arowmajor_bskinny_m1n1(
  const float16_t *a_ptr, const float16_t *b_ptr, float16_t *c_ptr,
  uint32_t k_left, uint32_t LDK, uint32_t LDM,
  float16_t beta, bool c_rowmajor) {

  float16x8_t cq1;
  __asm__ __volatile__ (
    "movi %[cq1].16b,#0; movi v0.16b,#0\n\t"
    "mov v1.16b,%[cq1].16b; mov v2.16b,v0.16b\n\t"
    "cmp %w[k_left],#32; b.lt 3f\n\t"
    "ldr q3,[%[a_ptr]],#64; ldr q7,[%[b_ptr]],#64\n\t"
    "ldr q4,[%[a_ptr],#-48]; ldr q8,[%[b_ptr],#-48]\n\t"
    "ldr q5,[%[a_ptr],#-32]; ldr q9,[%[b_ptr],#-32]\n\t"
    "ldr q6,[%[a_ptr],#-16]; ldr q10,[%[b_ptr],#-16]\n\t"
    "cmp %w[k_left],#64; b.lt 2f\n\t"
    ".balign 16; 1:\n\t"
    "fmla %[cq1].8h,v3.8h,v7.8h; ldr q3,[%[a_ptr]],#64\n\t"
    "ldr q7,[%[b_ptr]],#64; sub %w[k_left],%w[k_left],#32\n\t"
    "fmla v0.8h,v4.8h,v8.8h; ldr q4,[%[a_ptr],#-48]\n\t"
    "ldr q8,[%[b_ptr],#-48]; cmp %w[k_left],#64\n\t"
    "fmla v1.8h,v5.8h,v9.8h; ldr q5,[%[a_ptr],#-32]\n\t"
    "ldr q9,[%[b_ptr],#-32]\n\t"
    "fmla v2.8h,v6.8h,v10.8h; ldr q6,[%[a_ptr],#-16]\n\t"
    "ldr q10,[%[b_ptr],#-16]; b.ge 1b\n\t"
    "2:\n\t"
    "fmla %[cq1].8h,v3.8h,v7.8h; sub %w[k_left],%w[k_left],#32\n\t"
    "fmla v0.8h,v4.8h,v8.8h\n\t"
    "fmla v1.8h,v5.8h,v9.8h\n\t"
    "fmla v2.8h,v6.8h,v10.8h\n\t"
    "3:\n\t"
    "cmp %w[k_left],#16; fadd %[cq1].8h,%[cq1].8h,v1.8h\n\t"
    "fadd v0.8h,v0.8h,v2.8h; b.lt 4f\n\t"
    "ldr q3,[%[a_ptr]],#32; ldr q7,[%[b_ptr]],#32\n\t"
    "ldr q4,[%[a_ptr],#-16]; ldr q8,[%[b_ptr],#-16]\n\t"
    "sub %w[k_left],%w[k_left],#16\n\t"
    "fmla %[cq1].8h,v3.8h,v7.8h; fmla v0.8h,v4.8h,v8.8h\n\t"
    "4:\n\t"
    "cmp %w[k_left],#8; fadd %[cq1].8h,%[cq1].8h,v0.8h; b.lt 5f\n\t"
    "ldr q3,[%[a_ptr]],#16; ldr q7,[%[b_ptr]],#16\n\t"
    "sub %w[k_left],%w[k_left],#8; fmla %[cq1].8h,v3.8h,v7.8h\n\t"
    "5:\n\t"
   :[cq1]"=w"(cq1), [k_left]"+r"(k_left),
    [a_ptr]"+r"(a_ptr), [b_ptr]"+r"(b_ptr)
   ::"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10");

  float16x4_t cd1 = vget_low_f16(vpaddq_f16(cq1, cq1));
  if (k_left > 3) {
    float16x4_t ad1 = vld1_f16(a_ptr); a_ptr += 4;
    float16x4_t bd1 = vld1_f16(b_ptr); b_ptr += 4;
    cd1 = vfma_f16(cd1, ad1, bd1); k_left -= 4;
  }

  float16_t cs1 = vget_lane_f16(cd1, 0) + vget_lane_f16(cd1, 1) +
    vget_lane_f16(cd1, 2) + vget_lane_f16(cd1, 3);
  for (; k_left > 0; k_left--) {
    float16_t as1 = *a_ptr; a_ptr++;
    float16_t bs1 = *b_ptr; b_ptr++;
    cs1 += as1 * bs1;
  }

  *c_ptr = c_ptr[0] * beta + cs1;
}

/* k_mask = 15 */
static inline void inline_hgemm_arowmajor_bskinny_m1n2(
  const float16_t *a_ptr, const float16_t *b_ptr, float16_t *c_ptr,
  uint32_t k_left, uint32_t LDK, uint32_t LDM,
  float16_t beta, bool c_rowmajor) {

  float16x8_t cq1, cq2;
  __asm__ __volatile__ (
    "movi %[cq1].16b,#0; movi %[cq2].16b,#0\n\t"
    "mov v0.16b,%[cq1].16b; mov v1.16b,%[cq2].16b\n\t"
    "cmp %w[k_left],#16; b.lt 3f\n\t"
    "ldr q2,[%[a_ptr]],#32; ldr q4,[%[b_ptr]],#64; ldr q6,[%[b_ptr],#-48]\n\t"
    "ldr q3,[%[a_ptr],#-16]; ldr q5,[%[b_ptr],#-32]; ldr q7,[%[b_ptr],#-16]\n\t"
    "cmp %w[k_left],#32; b.lt 2f\n\t"
    "1:\n\t"
    "fmla %[cq1].8h,v2.8h,v4.8h; ldr q4,[%[b_ptr]],#64\n\t"
    "sub %w[k_left],%w[k_left],#16\n\t"
    "fmla %[cq2].8h,v2.8h,v6.8h; ldr q6,[%[b_ptr],#-48]\n\t"
    "ldr q2,[%[a_ptr]],#32\n\t"
    "fmla v0.8h,v3.8h,v5.8h; ldr q5,[%[b_ptr],#-32]\n\t"
    "cmp %w[k_left],#32\n\t"
    "fmla v1.8h,v3.8h,v7.8h; ldr q7,[%[b_ptr],#-16]\n\t"
    "ldr q3,[%[a_ptr],#-16]\n\t"
    "b.ge 1b\n\t"
    "2:\n\t"
    "fmla %[cq1].8h,v2.8h,v4.8h; sub %w[k_left],%w[k_left],#16\n\t"
    "fmla %[cq2].8h,v2.8h,v6.8h\n\t"
    "fmla v0.8h,v3.8h,v5.8h\n\t"
    "fmla v1.8h,v3.8h,v7.8h\n\t"
    "3:\n\t"
    "cmp %w[k_left],#8; fadd %[cq1].8h,%[cq1].8h,v0.8h\n\t"
    "fadd %[cq2].8h,%[cq2].8h,v1.8h; b.lt 4f\n\t"
    "ldr q2,[%[a_ptr]],#16; ldr q4,[%[b_ptr]],#32; ldr q6,[%[b_ptr],#-16]\n\t"
    "sub %w[k_left],%w[k_left],#8\n\t"
    "fmla %[cq1].8h,v2.8h,v4.8h; fmla %[cq2].8h,v2.8h,v6.8h\n\t"
    "4:\n\t"
   :[cq1]"=w"(cq1), [cq2]"=w"(cq2), [k_left]"+r"(k_left),
    [a_ptr]"+r"(a_ptr), [b_ptr]"+r"(b_ptr)
   ::"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7");

  cq1 = vpaddq_f16(cq1, cq2);
  if (k_left > 3) {
    float16x4_t ad1 = vld1_f16(a_ptr); a_ptr += 4;
    float16x8_t aq1 = vcombine_f16(ad1, ad1);
    float16x8_t bq1 = vld1q_f16(b_ptr); b_ptr += 8;
    cq1 = vfmaq_f16(cq1, aq1, bq1); k_left -= 4;
  }

  const float16x8_t cz1 = vdupq_n_f16(0);
  float16x4_t cd1 = vget_low_f16(vpaddq_f16(cq1, cz1));
  if (k_left > 1) {
    float16x4_t ad1;
    __asm__("ld1r {%0.2s},[%1],#4":"=w"(ad1),"+r"(a_ptr)::"memory");
    float16x4_t bd1 = vld1_f16(b_ptr); b_ptr += 4;
    cd1 = vfma_f16(cd1, ad1, bd1); k_left -= 2;
  }
  
  cd1 = vpadd_f16(cd1, vget_low_f16(cz1));
  if (k_left > 0) {
    float16x4_t ad1, bd1;
    __asm__("ld1r {%0.4h},[%1],#2":"=w"(ad1),"+r"(a_ptr)::"memory");
    __asm__("ldr %s0,[%1],#4":"=w"(bd1),"+r"(b_ptr)::"memory");
    cd1 = vfma_f16(cd1, ad1, bd1);
  }
  
  if (c_rowmajor) {
    c_ptr[0] = c_ptr[0] * beta + vget_lane_f16(cd1, 0);
    c_ptr[1] = c_ptr[1] * beta + vget_lane_f16(cd1, 1);
  } else {
    c_ptr[0] = c_ptr[0] * beta + vget_lane_f16(cd1, 0);
    c_ptr[LDM] = c_ptr[LDM] * beta + vget_lane_f16(cd1, 1);
  }
}

/* k_mask = 13 */
static inline void inline_hgemm_arowmajor_bskinny_m1n3(
  const float16_t *a_ptr, const float16_t *b_ptr, float16_t *c_ptr,
  uint32_t k_left, uint32_t LDK, uint32_t LDM,
  float16_t beta, bool c_rowmajor) {

  float16x8_t cq1, cq2, cq3;
  __asm__ __volatile__ (
    "movi %[cq1].16b,#0; movi %[cq2].16b,#0; movi %[cq3].16b,#0\n\t"
    "mov v0.16b,%[cq1].16b; mov v1.16b,%[cq2].16b; mov v2.16b,%[cq3].16b\n\t"
    "cmp %w[k_left],#16; b.lt 3f\n\t"
    "ldr q3,[%[a_ptr]],#32; ldr q5,[%[b_ptr]],#96\n\t"
    "ldr q7,[%[b_ptr],#-80]; ldr q9,[%[b_ptr],#-64]\n\t"
    "ldr q4,[%[a_ptr],#-16]; ldr q6,[%[b_ptr],#-48]\n\t"
    "ldr q8,[%[b_ptr],#-32]; ldr q10,[%[b_ptr],#-16]\n\t"
    "cmp %w[k_left],#32; b.lt 2f\n\t"
    "1:\n\t"
    "fmla %[cq1].8h,v3.8h,v5.8h; ldr q5,[%[b_ptr]],#96\n\t"
    "sub %w[k_left],%w[k_left],#16\n\t"
    "fmla %[cq2].8h,v3.8h,v7.8h; ldr q7,[%[b_ptr],#-80]\n\t"
    "fmla %[cq3].8h,v3.8h,v9.8h; ldr q9,[%[b_ptr],#-64]\n\t"
    "ldr q3,[%[a_ptr]],#32\n\t"
    "fmla v0.8h,v4.8h,v6.8h; ldr q6,[%[b_ptr],#-48]\n\t"
    "cmp %w[k_left],#32\n\t"
    "fmla v1.8h,v4.8h,v8.8h; ldr q8,[%[b_ptr],#-32]\n\t"
    "fmla v2.8h,v4.8h,v10.8h; ldr q10,[%[b_ptr],#-16]\n\t"
    "ldr q4,[%[a_ptr],#-16]\n\t"
    "b.ge 1b\n\t"
    "2:\n\t"
    "fmla %[cq1].8h,v3.8h,v5.8h; sub %w[k_left],%w[k_left],#16\n\t"
    "fmla %[cq2].8h,v3.8h,v7.8h\n\t"
    "fmla %[cq3].8h,v3.8h,v9.8h\n\t"
    "fmla v0.8h,v4.8h,v6.8h\n\t"
    "fmla v1.8h,v4.8h,v8.8h\n\t"
    "fmla v2.8h,v4.8h,v10.8h\n\t"
    "3:\n\t"
    "cmp %w[k_left],#8\n\t"
    "fadd %[cq1].8h,%[cq1].8h,v0.8h\n\t"
    "fadd %[cq2].8h,%[cq2].8h,v1.8h\n\t"
    "fadd %[cq3].8h,%[cq3].8h,v2.8h; b.lt 4f\n\t"
    "ldr q3,[%[a_ptr]],#16; ldr q5,[%[b_ptr]],#48\n\t"
    "ldr q7,[%[b_ptr],#-32]; ldr q9,[%[b_ptr],#-16]\n\t"
    "sub %w[k_left],%w[k_left],#8\n\t"
    "fmla %[cq1].8h,v3.8h,v5.8h\n\t"
    "fmla %[cq2].8h,v3.8h,v7.8h\n\t"
    "fmla %[cq3].8h,v3.8h,v9.8h\n\t"
    "4:\n\t"
   :[cq1]"=w"(cq1), [cq2]"=w"(cq2), [cq3]"=w"(cq3),
    [k_left]"+r"(k_left), [a_ptr]"+r"(a_ptr), [b_ptr]"+r"(b_ptr)
   ::"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10");

  float16x4_t cd1 = vadd_f16(vget_low_f16(cq1), vget_high_f16(cq1));
  float16x4_t cd2 = vadd_f16(vget_low_f16(cq2), vget_high_f16(cq2));
  float16x4_t cd3 = vadd_f16(vget_low_f16(cq3), vget_high_f16(cq3));
  if (k_left > 3) {
    float16x4_t ad1 = vld1_f16(a_ptr); a_ptr += 4;
    float16x4_t bd1 = vld1_f16(b_ptr);
    float16x4_t bd2 = vld1_f16(b_ptr + 4);
    float16x4_t bd3 = vld1_f16(b_ptr + 8); b_ptr += 12;
    cd1 = vfma_f16(cd1, ad1, bd1);
    cd2 = vfma_f16(cd2, ad1, bd2);
    cd3 = vfma_f16(cd3, ad1, bd3); k_left -= 4;
  }

  float16_t cs1 = vget_lane_f16(cd1, 0) + vget_lane_f16(cd1, 1) +
    vget_lane_f16(cd1, 2) + vget_lane_f16(cd1, 3);
  float16_t cs2 = vget_lane_f16(cd2, 0) + vget_lane_f16(cd2, 1) +
    vget_lane_f16(cd2, 2) + vget_lane_f16(cd2, 3);
  float16_t cs3 = vget_lane_f16(cd3, 0) + vget_lane_f16(cd3, 1) +
    vget_lane_f16(cd3, 2) + vget_lane_f16(cd3, 3);
  for (; k_left > 0; k_left--) {
    float16_t as1 = *a_ptr; a_ptr++;
    cs1 += as1 * b_ptr[0];
    cs2 += as1 * b_ptr[1];
    cs3 += as1 * b_ptr[2]; b_ptr += 3;
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

typedef float16_t hgemm_skinnydot_ascalar;
typedef float16_t hgemm_skinnydot_bscalar;
typedef float16_t hgemm_skinnydot_cscalar;

static inline bool unroll_test_m1n1(uint32_t M, uint32_t K) {
  return true;
}

static inline bool unroll_test_m1n2(uint32_t M, uint32_t K) {
  return true;
}

static inline bool unroll_test_m1n3(uint32_t M, uint32_t K) {
  return true;
}

GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(hgemm, 1, 13, 1, 65536, float16_t, float16_t, unroll_test)
GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(hgemm, 2, 15, 1, 65536, float16_t, float16_t, unroll_test)
GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(hgemm, 3, 13, 1, 65536, float16_t, float16_t, unroll_test)

typedef float16_t hgemm_skinnydot_avec1;
typedef float16_t hgemm_skinnydot_bvec1;
typedef float16_t hgemm_skinnydot_cvec1;

typedef float16x4_t hgemm_skinnydot_avec4;
typedef float16x4_t hgemm_skinnydot_bvec4;
typedef float16x4_t hgemm_skinnydot_cvec4;

typedef float16x8_t hgemm_skinnydot_avec8;
typedef float16x8_t hgemm_skinnydot_bvec8;
typedef float16x8_t hgemm_skinnydot_cvec8;

GEMM_SKINNY_DOT_CALC_UNIT(hgemm, 8) {
  return vfmaq_f16(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_DOT_CALC_UNIT(hgemm, 4) {
  return vfma_f16(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_DOT_CALC_UNIT(hgemm, 1) {
  return c_vec + a_vec * b_vec;
}

GEMM_SKINNY_DOT_LOADA_UNIT(hgemm, 8) {
  __asm__("prfm pldl1keep,[%0,#80]"::"r"(a_ptr):);
  return vld1q_f16(a_ptr);
}

GEMM_SKINNY_DOT_LOADA_UNIT(hgemm, 4) {
  __asm__("prfm pldl1keep,[%0,#72]"::"r"(a_ptr):);
  return vld1_f16(a_ptr);
}

GEMM_SKINNY_DOT_LOADA_UNIT(hgemm, 1) {
  return *a_ptr;
}

GEMM_SKINNY_DOT_LOADB_UNIT(hgemm, 8) {
  return vld1q_f16(b_ptr);
}

GEMM_SKINNY_DOT_LOADB_UNIT(hgemm, 4) {
  return vld1_f16(b_ptr);
}

GEMM_SKINNY_DOT_LOADB_UNIT(hgemm, 1) {
  return *b_ptr;
}

GEMM_SKINNY_DOT_REDUC_UNIT(hgemm, 8, 4) {
  return vget_low_f16(vpaddq_f16(c_vec, c_vec));
}

GEMM_SKINNY_DOT_REDUC_UNIT(hgemm, 4, 1) {
  float cs1 = vget_lane_f16(c_vec, 0);
  float cs2 = vget_lane_f16(c_vec, 1);
  float cs3 = vget_lane_f16(c_vec, 2);
  float cs4 = vget_lane_f16(c_vec, 3);
  cs1 += cs2; cs3 += cs4;
  return cs1 + cs3;
}

GEMM_SKINNY_DOT_INITC_UNIT(hgemm, 8) {
  return vdupq_n_f16(0);
}

GEMM_SKINNY_DOT_INITC_UNIT(hgemm, 4) {
  return vdup_n_f16(0);
}

GEMM_SKINNY_DOT_INITC_UNIT(hgemm, 1) {
  return 0;
}

GEMM_SKINNY_DOT_PARALLEL_FUNC(hgemm, 4, 13, 7, 65536, float16_t, float16_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(hgemm, 5, 13, 7, 65536, float16_t, float16_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(hgemm, 6, 13, 7, 65536, float16_t, float16_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(hgemm, 7, 13, 3, 65536, float16_t, float16_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(hgemm, 8, 13, 3, 65536, float16_t, float16_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(hgemm, 9, 13, 3, 65536, float16_t, float16_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(hgemm, 10, 13, 3, 65536, float16_t, float16_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(hgemm, 11, 13, 3, 65536, float16_t, float16_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(hgemm, 12, 13, 3, 65536, float16_t, float16_t)
