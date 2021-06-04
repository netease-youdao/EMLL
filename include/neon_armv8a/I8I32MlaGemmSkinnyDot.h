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


#include "arm_neon/NeonI8I32MlaGemmSkinnyDot.h"

#ifndef INCLUDE_ARMV8_I8I32_SKINNYDOT_ASM
#define INCLUDE_ARMV8_I8I32_SKINNYDOT_ASM

#define I8I32MLA_SKINNYDOT_INLINE_M4N1(gemm) \
static inline void inline_##gemm##_arowmajor_bskinny_m4n1(\
  const I8 *a_ptr1, const I8 *b_ptr, I32 *c_ptr,\
  uint32_t k_left, uint32_t LDK, uint32_t LDM,\
  I32 beta, bool c_rowmajor) {\
\
  const I8 *a_ptr2 = a_ptr1 + LDK;\
  const I8 *a_ptr3 = a_ptr1 + LDK * 2;\
  const I8 *a_ptr4 = a_ptr2 + LDK * 2;\
  I32X2 cd1, cd2;\
  const uint32_t next_pref = (LDK * 4 - k_left) + 16;\
  __asm__ __volatile__ (\
    "movi v8.16b,#0; movi v9.16b,#0; movi v10.16b,#0; movi v11.16b,#0\n\t"\
    "cmp %w[k_left],#16; b.lt 3f\n\t"\
    "ldr q0,[%[a_ptr1]],#16; ldr q1,[%[a_ptr2]],#16\n\t"\
    "ldr q2,[%[a_ptr3]],#16; ldr q3,[%[a_ptr4]],#16\n\t"\
    "ldr q4,[%[b_ptr]],#16\n\t"\
    "cmp %w[k_left],#32; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    ""IMULL" v12.8h,v0.8b,v4.8b; prfm pldl1keep,[%[a_ptr1],#64]\n\t"\
    ""IMULL" v13.8h,v1.8b,v4.8b; prfm pldl1keep,[%[a_ptr2],#64]\n\t"\
    ""IMULL" v14.8h,v2.8b,v4.8b; prfm pldl1keep,[%[a_ptr3],#64]\n\t"\
    ""IMULL" v15.8h,v3.8b,v4.8b; prfm pldl1keep,[%[a_ptr4],#64]\n\t"\
    ""IADALP" v8.4s,v12.8h; "IMULL"2 v12.8h,v0.16b,v4.16b\n\t"\
    "ldr q0,[%[a_ptr1]],#16\n\t"\
    ""IADALP" v9.4s,v13.8h; "IMULL"2 v13.8h,v1.16b,v4.16b\n\t"\
    "ldr q1,[%[a_ptr2]],#16\n\t"\
    ""IADALP" v10.4s,v14.8h; "IMULL"2 v14.8h,v2.16b,v4.16b\n\t"\
    "ldr q2,[%[a_ptr3]],#16\n\t"\
    ""IADALP" v11.4s,v15.8h; "IMULL"2 v15.8h,v3.16b,v4.16b\n\t"\
    "ldr q3,[%[a_ptr4]],#16\n\t"\
    "ldr q4,[%[b_ptr]],#16\n\t"\
    ""IADALP" v8.4s,v12.8h; sub %w[k_left],%w[k_left],#16\n\t"\
    ""IADALP" v9.4s,v13.8h\n\t"\
    ""IADALP" v10.4s,v14.8h; cmp %w[k_left],#32\n\t"\
    ""IADALP" v11.4s,v15.8h; b.ge 1b\n\t"\
    "2:\n\t"\
    ""IMULL" v12.8h,v0.8b,v4.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr1],%w[next_pref],SXTW #0]\n\t"\
    ""IMULL" v13.8h,v1.8b,v4.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr2],%w[next_pref],SXTW #0]\n\t"\
    ""IMULL" v14.8h,v2.8b,v4.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr3],%w[next_pref],SXTW #0]\n\t"\
    ""IMULL" v15.8h,v3.8b,v4.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr4],%w[next_pref],SXTW #0]\n\t"\
    ""IADALP" v8.4s,v12.8h; "IMULL"2 v12.8h,v0.16b,v4.16b\n\t"\
    ""IADALP" v9.4s,v13.8h; "IMULL"2 v13.8h,v1.16b,v4.16b\n\t"\
    ""IADALP" v10.4s,v14.8h; "IMULL"2 v14.8h,v2.16b,v4.16b\n\t"\
    ""IADALP" v11.4s,v15.8h; "IMULL"2 v15.8h,v3.16b,v4.16b\n\t"\
    ""IADALP" v8.4s,v12.8h; "IADALP" v9.4s,v13.8h\n\t"\
    "sub %w[k_left],%w[k_left],#16\n\t"\
    ""IADALP" v10.4s,v14.8h; "IADALP" v11.4s,v15.8h\n\t"\
    "3:\n\t"\
    "cmp %w[k_left],#8; b.lt 4f\n\t"\
    "ldr d0,[%[a_ptr1]],#8; ldr d1,[%[a_ptr2]],#8\n\t"\
    "ldr d2,[%[a_ptr3]],#8; ldr d3,[%[a_ptr4]],#8\n\t"\
    "ldr d4,[%[b_ptr]],#8; sub %w[k_left],%w[k_left],#8\n\t"\
    ""IMULL" v12.8h,v0.8b,v4.8b; "IMULL" v13.8h,v1.8b,v4.8b\n\t"\
    ""IMULL" v14.8h,v2.8b,v4.8b; "IMULL" v15.8h,v3.8b,v4.8b\n\t"\
    ""IADALP" v8.4s,v12.8h; "IADALP" v9.4s,v13.8h\n\t"\
    ""IADALP" v10.4s,v14.8h; "IADALP" v11.4s,v15.8h\n\t"\
    "4:\n\t"\
    "movi v12.16b,#0\n\t"\
    "addp v8.4s,v8.4s,v12.4s; addp v9.4s,v9.4s,v12.4s\n\t"\
    "addp v10.4s,v10.4s,v12.4s; addp v11.4s,v11.4s,v12.4s\n\t"\
    "cmp %w[k_left],#4; b.lt 5f\n\t"\
    "ldr s0,[%[a_ptr1]],#4; ldr s1,[%[a_ptr2]],#4\n\t"\
    "ldr s2,[%[a_ptr3]],#4; ldr s3,[%[a_ptr4]],#4\n\t"\
    "ldr s4,[%[b_ptr]],#4; sub %w[k_left],%w[k_left],#4\n\t"\
    ""IMULL" v12.8h,v0.8b,v4.8b; "IMULL" v13.8h,v1.8b,v4.8b\n\t"\
    ""IMULL" v14.8h,v2.8b,v4.8b; "IMULL" v15.8h,v3.8b,v4.8b\n\t"\
    ""IADALP" v8.2s,v12.4h; "IADALP" v9.2s,v13.4h\n\t"\
    ""IADALP" v10.2s,v14.4h; "IADALP" v11.2s,v15.4h\n\t"\
    "5:\n\t"\
    "cmp %w[k_left],#2; b.lt 6f\n\t"\
    "ldr h0,[%[a_ptr1]],#2; ldr h1,[%[a_ptr2]],#2\n\t"\
    "ldr h2,[%[a_ptr3]],#2; ldr h3,[%[a_ptr4]],#2\n\t"\
    "ldr h4,[%[b_ptr]],#2; sub %w[k_left],%w[k_left],#2\n\t"\
    ""IXTL" v0.8h,v0.8b; "IXTL" v1.8h,v1.8b\n\t"\
    ""IXTL" v2.8h,v2.8b; "IXTL" v3.8h,v3.8b; "IXTL" v4.8h,v4.8b\n\t"\
    ""IMLAL" v8.4s,v0.4h,v4.4h; "IMLAL" v9.4s,v1.4h,v4.4h\n\t"\
    ""IMLAL" v10.4s,v2.4h,v4.4h; "IMLAL" v11.4s,v3.4h,v4.4h\n\t"\
    "6:\n\t"\
    "addp %[cd1].2s,v8.2s,v9.2s; addp %[cd2].2s,v10.2s,v11.2s\n\t"\
    "cmp %w[k_left],#1; b.lt 7f\n\t"\
    "ldr b0,[%[a_ptr1]],#1; ldr b1,[%[a_ptr2]],#1\n\t"\
    "ldr b2,[%[a_ptr3]],#1; ldr b3,[%[a_ptr4]],#1\n\t"\
    "ldr b4,[%[b_ptr]],#1; sub %w[k_left],%w[k_left],#1\n\t"\
    "ins v0.b[1],v1.b[0]; ins v2.b[1],v3.b[0]\n\t"\
    ""IXTL" v0.8h,v0.8b; "IXTL" v2.8h,v2.8b; "IXTL" v4.8h,v4.8b\n\t"\
    ""IMLAL" %[cd1].4s,v0.4h,v4.h[0]; "IMLAL" %[cd2].4s,v2.4h,v4.h[0]\n\t"\
    "7:\n\t"\
   :[cd1]"=w"(cd1), [cd2]"=w"(cd2), [k_left]"+r"(k_left), [b_ptr]"+r"(b_ptr),\
    [a_ptr1]"+r"(a_ptr1), [a_ptr2]"+r"(a_ptr2),\
    [a_ptr3]"+r"(a_ptr3), [a_ptr4]"+r"(a_ptr4)\
   :[next_pref]"r"(next_pref)\
   :"cc","memory","v0","v1","v2","v3","v4",\
    "v8","v9","v10","v11","v12","v13","v14","v15");\
\
  cd1 = VMLA_N_I32(cd1, VLD1_I32(c_ptr), beta);\
  cd2 = VMLA_N_I32(cd2, VLD1_I32(c_ptr + 2), beta);\
  VST1_I32(c_ptr, cd1);\
  VST1_I32(c_ptr + 2, cd2);\
}

/* k_mask = 31 */
#define I8I32MLA_SKINNYDOT_INLINE_M4N2(gemm) \
static inline void inline_##gemm##_arowmajor_bskinny_m4n2(\
  const I8 *a_ptr1, const I8 *b_ptr, I32 *c_ptr,\
  uint32_t k_left, uint32_t LDK, uint32_t LDM,\
  I32 beta, bool c_rowmajor) {\
\
  const I8 *a_ptr2 = a_ptr1 + LDK;\
  const I8 *a_ptr3 = a_ptr1 + LDK * 2;\
  const I8 *a_ptr4 = a_ptr2 + LDK * 2;\
  I32X4 cq1, cq2, cq3, cq4; /* higher 2 elements not used */\
  const uint32_t next_pref = (LDK * 4 - k_left) + 16;\
  __asm__ __volatile__ (\
    "movi v6.16b,#0; movi v7.16b,#0\n\t"\
    "movi v8.16b,#0; movi v9.16b,#0\n\t"\
    "movi v10.16b,#0; movi v11.16b,#0\n\t"\
    "movi v12.16b,#0; movi v13.16b,#0\n\t"\
    "cmp %w[k_left],#16; b.lt 3f\n\t"\
    "ldr q0,[%[a_ptr1]],#16; ldr q1,[%[a_ptr2]],#16\n\t"\
    "ldr q2,[%[a_ptr3]],#16; ldr q3,[%[a_ptr4]],#16\n\t"\
    "ldr q4,[%[b_ptr]]; ldr q5,[%[b_ptr],#16]; add %[b_ptr],%[b_ptr],#32\n\t"\
    "cmp %w[k_left],#32; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    ""IMULL" v14.8h,v0.8b,v4.8b; "IMULL" v18.8h,v0.8b,v5.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr1],#64]\n\t"\
    ""IMULL" v15.8h,v1.8b,v4.8b; "IMULL" v19.8h,v1.8b,v5.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr2],#64]\n\t"\
    ""IMULL" v16.8h,v2.8b,v4.8b; "IMULL" v20.8h,v2.8b,v5.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr3],#64]\n\t"\
    ""IMULL" v17.8h,v3.8b,v4.8b; "IMULL" v21.8h,v3.8b,v5.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr4],#64]\n\t"\
    ""IADALP" v6.4s,v14.8h; "IMULL"2 v14.8h,v0.16b,v4.16b\n\t"\
    ""IADALP" v10.4s,v18.8h; "IMULL"2 v18.8h,v0.16b,v5.16b\n\t"\
    "ldr q0,[%[a_ptr1]],#16\n\t"\
    ""IADALP" v7.4s,v15.8h; "IMULL"2 v15.8h,v1.16b,v4.16b\n\t"\
    ""IADALP" v11.4s,v19.8h; "IMULL"2 v19.8h,v1.16b,v5.16b\n\t"\
    "ldr q1,[%[a_ptr2]],#16\n\t"\
    ""IADALP" v8.4s,v16.8h; "IMULL"2 v16.8h,v2.16b,v4.16b\n\t"\
    ""IADALP" v12.4s,v20.8h; "IMULL"2 v20.8h,v2.16b,v5.16b\n\t"\
    "ldr q2,[%[a_ptr3]],#16\n\t"\
    ""IADALP" v9.4s,v17.8h; "IMULL"2 v17.8h,v3.16b,v4.16b\n\t"\
    ""IADALP" v13.4s,v21.8h; "IMULL"2 v21.8h,v3.16b,v5.16b\n\t"\
    "ldr q3,[%[a_ptr4]],#16\n\t"\
    ""IADALP" v6.4s,v14.8h; "IADALP" v10.4s,v18.8h\n\t"\
    "ldr q4,[%[b_ptr]],#32\n\t"\
    ""IADALP" v7.4s,v15.8h; "IADALP" v11.4s,v19.8h\n\t"\
    "ldr q5,[%[b_ptr],#-16]\n\t"\
    ""IADALP" v8.4s,v16.8h; sub %w[k_left],%w[k_left],#16\n\t"\
    ""IADALP" v12.4s,v20.8h; cmp %w[k_left],#32\n\t"\
    ""IADALP" v9.4s,v17.8h; "IADALP" v13.4s,v21.8h\n\t"\
    "b.ge 1b\n\t"\
    "2:\n\t"\
    ""IMULL" v14.8h,v0.8b,v4.8b; "IMULL" v18.8h,v0.8b,v5.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr1],%w[next_pref],SXTW #0]\n\t"\
    ""IMULL" v15.8h,v1.8b,v4.8b; "IMULL" v19.8h,v1.8b,v5.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr2],%w[next_pref],SXTW #0]\n\t"\
    ""IMULL" v16.8h,v2.8b,v4.8b; "IMULL" v20.8h,v2.8b,v5.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr3],%w[next_pref],SXTW #0]\n\t"\
    ""IMULL" v17.8h,v3.8b,v4.8b; "IMULL" v21.8h,v3.8b,v5.8b\n\t"\
    "prfm pldl1keep,[%[a_ptr4],%w[next_pref],SXTW #0]\n\t"\
    ""IADALP" v6.4s,v14.8h; "IMULL"2 v14.8h,v0.16b,v4.16b\n\t"\
    ""IADALP" v10.4s,v18.8h; "IMULL"2 v18.8h,v0.16b,v5.16b\n\t"\
    ""IADALP" v7.4s,v15.8h; "IMULL"2 v15.8h,v1.16b,v4.16b\n\t"\
    ""IADALP" v11.4s,v19.8h; "IMULL"2 v19.8h,v1.16b,v5.16b\n\t"\
    ""IADALP" v8.4s,v16.8h; "IMULL"2 v16.8h,v2.16b,v4.16b\n\t"\
    ""IADALP" v12.4s,v20.8h; "IMULL"2 v20.8h,v2.16b,v5.16b\n\t"\
    ""IADALP" v9.4s,v17.8h; "IMULL"2 v17.8h,v3.16b,v4.16b\n\t"\
    ""IADALP" v13.4s,v21.8h; "IMULL"2 v21.8h,v3.16b,v5.16b\n\t"\
    ""IADALP" v6.4s,v14.8h; "IADALP" v10.4s,v18.8h\n\t"\
    ""IADALP" v7.4s,v15.8h; "IADALP" v11.4s,v19.8h\n\t"\
    ""IADALP" v8.4s,v16.8h; sub %w[k_left],%w[k_left],#16\n\t"\
    ""IADALP" v12.4s,v20.8h\n\t"\
    ""IADALP" v9.4s,v17.8h; "IADALP" v13.4s,v21.8h\n\t"\
    "3:\n\t"\
    "cmp %w[k_left],#8; b.lt 4f\n\t"\
    "ldr d0,[%[a_ptr1]],#8; ldr d1,[%[a_ptr2]],#8\n\t"\
    "ldr d2,[%[a_ptr3]],#8; ldr d3,[%[a_ptr4]],#8\n\t"\
    "ldr d4,[%[b_ptr]],#16; ldr d5,[%[b_ptr],#-8]\n\t"\
    ""IMULL" v14.8h,v0.8b,v4.8b; "IMULL" v18.8h,v0.8b,v5.8b\n\t"\
    ""IMULL" v15.8h,v1.8b,v4.8b; "IMULL" v19.8h,v1.8b,v5.8b\n\t"\
    ""IMULL" v16.8h,v2.8b,v4.8b; "IMULL" v20.8h,v2.8b,v5.8b\n\t"\
    ""IMULL" v17.8h,v3.8b,v4.8b; "IMULL" v21.8h,v3.8b,v5.8b\n\t"\
    "sub %w[k_left],%w[k_left],#8\n\t"\
    ""IADALP" v6.4s,v14.8h; "IADALP" v10.4s,v18.8h\n\t"\
    ""IADALP" v7.4s,v15.8h; "IADALP" v11.4s,v19.8h\n\t"\
    ""IADALP" v8.4s,v16.8h; "IADALP" v12.4s,v20.8h\n\t"\
    ""IADALP" v9.4s,v17.8h; "IADALP" v13.4s,v21.8h\n\t"\
    "4:\n\t"\
    "addp v6.4s,v6.4s,v10.4s; addp v7.4s,v7.4s,v11.4s\n\t"\
    "addp v8.4s,v8.4s,v12.4s; addp v9.4s,v9.4s,v13.4s\n\t"\
    "cmp %w[k_left],#4; b.lt 5f\n\t"\
    "ldr s4,[%[b_ptr]],#8; ldr s5,[%[b_ptr],#-4]\n\t"\
    "ld1r {v0.2s},[%[a_ptr1]],#4; ld1r {v1.2s},[%[a_ptr2]],#4\n\t"\
    "ins v4.s[1],v5.s[0]\n\t"\
    "ld1r {v2.2s},[%[a_ptr3]],#4; ld1r {v3.2s},[%[a_ptr4]],#4\n\t"\
    "sub %w[k_left],%w[k_left],#4\n\t"\
    ""IMULL" v14.8h,v0.8b,v4.8b; "IMULL" v15.8h,v1.8b,v4.8b\n\t"\
    ""IMULL" v16.8h,v2.8b,v4.8b; "IMULL" v17.8h,v3.8b,v4.8b\n\t"\
    ""IADALP" v6.4s,v14.8h; "IADALP" v7.4s,v15.8h\n\t"\
    ""IADALP" v8.4s,v16.8h; "IADALP" v9.4s,v17.8h\n\t"\
    "5:\n\t"\
    "movi v14.16b,#0\n\t"\
    "addp %[cq1].4s,v6.4s,v14.4s; addp %[cq2].4s,v7.4s,v14.4s\n\t"\
    "addp %[cq3].4s,v8.4s,v14.4s; addp %[cq4].4s,v9.4s,v14.4s\n\t"\
    "cmp %w[k_left],#2; b.lt 6f\n\t"\
    "ldr h4,[%[b_ptr]],#4; ldr h5,[%[b_ptr],#-2]\n\t"\
    "ld1r {v0.4h},[%[a_ptr1]],#2; ld1r {v1.4h},[%[a_ptr2]],#2\n\t"\
    "sub %w[k_left],%w[k_left],#2\n\t"\
    "ins v4.h[1],v5.h[0]\n\t"\
    "ld1r {v2.4h},[%[a_ptr3]],#2; ld1r {v3.4h},[%[a_ptr4]],#2\n\t"\
    ""IMULL" v14.8h,v0.8b,v4.8b; "IMULL" v15.8h,v1.8b,v4.8b\n\t"\
    ""IMULL" v16.8h,v2.8b,v4.8b; "IMULL" v17.8h,v3.8b,v4.8b\n\t"\
    ""IADALP" %[cq1].2s,v14.4h; "IADALP" %[cq2].2s,v15.4h\n\t"\
    ""IADALP" %[cq3].2s,v16.4h; "IADALP" %[cq4].2s,v17.4h\n\t"\
    "6:\n\t"\
    "cmp %w[k_left],#1; b.lt 7f\n\t"\
    "ldr b0,[%[a_ptr1]],#1; ldr b1,[%[a_ptr2]],#1\n\t"\
    "ldr b2,[%[a_ptr3]],#1; ldr b3,[%[a_ptr4]],#1\n\t"\
    "ldr b4,[%[b_ptr]],#2; ldr b5,[%[b_ptr],#-1]\n\t"\
    "ins v0.b[1],v1.b[0]; ins v2.b[1],v3.b[0]; ins v4.b[1],v5.b[0]\n\t"\
    ""IXTL" v0.8h,v0.8b; "IXTL" v2.8h,v2.8b; "IXTL" v4.8h,v4.8b\n\t"\
    "sub %w[k_left],%w[k_left],#1\n\t"\
    ""IMLAL" %[cq1].4s,v4.4h,v0.h[0]; "IMLAL" %[cq2].4s,v4.4h,v0.h[1]\n\t"\
    ""IMLAL" %[cq3].4s,v4.4h,v2.h[0]; "IMLAL" %[cq4].4s,v4.4h,v2.h[1]\n\t"\
    "7:\n\t"\
   :[cq1]"=w"(cq1), [cq2]"=w"(cq2), [cq3]"=w"(cq3), [cq4]"=w"(cq4),\
    [a_ptr1]"+r"(a_ptr1), [a_ptr2]"+r"(a_ptr2), [a_ptr3]"+r"(a_ptr3),\
    [a_ptr4]"+r"(a_ptr4), [b_ptr]"+r"(b_ptr),\
    [k_left]"+r"(k_left)\
   :[next_pref]"r"(next_pref)\
   :"cc","memory","v0","v1","v2","v3","v4","v5",\
    "v6","v7","v8","v9","v10","v11","v12","v13",\
    "v14","v15","v16","v17","v18","v19","v20","v21");\
\
  I32X2 cd1 = VGET_LOW_I32(cq1);\
  I32X2 cd2 = VGET_LOW_I32(cq2);\
  I32X2 cd3 = VGET_LOW_I32(cq3);\
  I32X2 cd4 = VGET_LOW_I32(cq4);\
  if (c_rowmajor) {\
    cd1 = VMLA_N_I32(cd1, VLD1_I32(c_ptr), beta);\
    cd2 = VMLA_N_I32(cd2, VLD1_I32(c_ptr + 2), beta);\
    cd3 = VMLA_N_I32(cd3, VLD1_I32(c_ptr + 4), beta);\
    cd4 = VMLA_N_I32(cd4, VLD1_I32(c_ptr + 6), beta);\
    VST1_I32(c_ptr, cd1);\
    VST1_I32(c_ptr + 2, cd2);\
    VST1_I32(c_ptr + 4, cd3);\
    VST1_I32(c_ptr + 6, cd4);\
  } else {\
    I32 *c_ptr2 = c_ptr + LDM;\
    I32X2 cdl1 = VZIP1_I32(cd1, cd2);\
    I32X2 cdl2 = VZIP1_I32(cd3, cd4);\
    I32X2 cdl3 = VZIP2_I32(cd1, cd2);\
    I32X2 cdl4 = VZIP2_I32(cd3, cd4);\
    cdl1 = VMLA_N_I32(cdl1, VLD1_I32(c_ptr), beta);\
    cdl2 = VMLA_N_I32(cdl2, VLD1_I32(c_ptr + 2), beta);\
    cdl3 = VMLA_N_I32(cdl3, VLD1_I32(c_ptr2), beta);\
    cdl4 = VMLA_N_I32(cdl4, VLD1_I32(c_ptr2 + 2), beta);\
    VST1_I32(c_ptr, cdl1); VST1_I32(c_ptr + 2, cdl2);\
    VST1_I32(c_ptr2, cdl3); VST1_I32(c_ptr2 + 2, cdl4);\
  }\
} 

/* k_mask = 31 */
#define I8I32MLA_SKINNYDOT_INLINE_M4N3(gemm) \
static inline void inline_##gemm##_arowmajor_bskinny_m4n3(\
  const I8 *a_ptr1, const I8 *b_ptr, I32 *c_ptr,\
  uint32_t k_left, uint32_t LDK, uint32_t LDM,\
  I32 beta, bool c_rowmajor) {\
\
  const I8 *a_ptr2 = a_ptr1 + LDK;\
  const I8 *a_ptr3 = a_ptr1 + LDK * 2;\
  const I8 *a_ptr4 = a_ptr2 + LDK * 2;\
  I32X4 cq1, cq2, cq3;\
  const uint32_t next_pref = (LDK * 4 - k_left) + 16;\
  __asm__ __volatile__ (\
    "movi %[q1].16b,#0; movi %[q2].16b,#0; movi %[q3].16b,#0\n\t"\
    "movi v10.16b,#0; movi v11.16b,#0; movi v12.16b,#0\n\t"\
    "movi v13.16b,#0; movi v14.16b,#0; movi v15.16b,#0\n\t"\
    "movi v16.16b,#0; movi v17.16b,#0; movi v18.16b,#0\n\t"\
    "cmp %w[k_left],#16; b.lt 3f\n\t"\
    "ldr q0,[%[a_ptr1]],#16; ldr q1,[%[a_ptr2]],#16\n\t"\
    "ldr q2,[%[a_ptr3]],#16; ldr q3,[%[a_ptr4]],#16\n\t"\
    "ldr q4,[%[b_ptr]]; ldr q5,[%[b_ptr],#16]\n\t"\
    "ldr q6,[%[b_ptr],#32]; add %[b_ptr],%[b_ptr],#48\n\t"\
    "cmp %w[k_left],#32; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    ""IMULL" v19.8h,v0.8b,v4.8b; "IMULL" v20.8h,v0.8b,v5.8b\n\t"\
    ""IMULL" v21.8h,v0.8b,v6.8b; prfm pldl1keep,[%[a_ptr1],#64]\n\t"\
    ""IMULL" v22.8h,v1.8b,v4.8b; "IMULL" v23.8h,v1.8b,v5.8b\n\t"\
    ""IMULL" v24.8h,v1.8b,v6.8b; prfm pldl1keep,[%[a_ptr2],#64]\n\t"\
    ""IMULL" v25.8h,v2.8b,v4.8b; "IMULL" v26.8h,v2.8b,v5.8b\n\t"\
    ""IMULL" v27.8h,v2.8b,v6.8b; prfm pldl1keep,[%[a_ptr3],#64]\n\t"\
    ""IMULL" v28.8h,v3.8b,v4.8b; "IMULL" v29.8h,v3.8b,v5.8b\n\t"\
    ""IMULL" v30.8h,v3.8b,v6.8b; prfm pldl1keep,[%[a_ptr4],#64]\n\t"\
    ""IADALP" %[q1].4s,v19.8h; "IMULL"2 v19.8h,v0.16b,v4.16b\n\t"\
    ""IADALP" %[q2].4s,v20.8h; "IMULL"2 v20.8h,v0.16b,v5.16b\n\t"\
    ""IADALP" %[q3].4s,v21.8h; "IMULL"2 v21.8h,v0.16b,v6.16b\n\t"\
    "ldr q0,[%[a_ptr1]],#16\n\t"\
    ""IADALP" v10.4s,v22.8h; "IMULL"2 v22.8h,v1.16b,v4.16b\n\t"\
    ""IADALP" v11.4s,v23.8h; "IMULL"2 v23.8h,v1.16b,v5.16b\n\t"\
    ""IADALP" v12.4s,v24.8h; "IMULL"2 v24.8h,v1.16b,v6.16b\n\t"\
    "ldr q1,[%[a_ptr2]],#16\n\t"\
    ""IADALP" v13.4s,v25.8h; "IMULL"2 v25.8h,v2.16b,v4.16b\n\t"\
    ""IADALP" v14.4s,v26.8h; "IMULL"2 v26.8h,v2.16b,v5.16b\n\t"\
    ""IADALP" v15.4s,v27.8h; "IMULL"2 v27.8h,v2.16b,v6.16b\n\t"\
    "ldr q2,[%[a_ptr3]],#16\n\t"\
    ""IADALP" v16.4s,v28.8h; "IMULL"2 v28.8h,v3.16b,v4.16b\n\t"\
    ""IADALP" v17.4s,v29.8h; "IMULL"2 v29.8h,v3.16b,v5.16b\n\t"\
    ""IADALP" v18.4s,v30.8h; "IMULL"2 v30.8h,v3.16b,v6.16b\n\t"\
    "ldr q3,[%[a_ptr4]],#16\n\t"\
    ""IADALP" %[q1].4s,v19.8h; "IADALP" %[q2].4s,v20.8h; "IADALP" %[q3].4s,v21.8h\n\t"\
    "ldr q4,[%[b_ptr]]; sub %w[k_left],%w[k_left],#16\n\t"\
    ""IADALP" v10.4s,v22.8h; "IADALP" v11.4s,v23.8h; "IADALP" v12.4s,v24.8h\n\t"\
    "ldr q5,[%[b_ptr],#16]\n\t"\
    ""IADALP" v13.4s,v25.8h; "IADALP" v14.4s,v26.8h; "IADALP" v15.4s,v27.8h\n\t"\
    "ldr q6,[%[b_ptr],#32]; add %[b_ptr],%[b_ptr],#48; cmp %w[k_left],#32\n\t"\
    ""IADALP" v16.4s,v28.8h; "IADALP" v17.4s,v29.8h; "IADALP" v18.4s,v30.8h\n\t"\
    "b.ge 1b\n\t"\
    "2:\n\t"\
    ""IMULL" v19.8h,v0.8b,v4.8b; "IMULL" v20.8h,v0.8b,v5.8b\n\t"\
    ""IMULL" v21.8h,v0.8b,v6.8b; prfm pldl1keep,[%[a_ptr1],%w[pref],SXTW #0]\n\t"\
    ""IMULL" v22.8h,v1.8b,v4.8b; "IMULL" v23.8h,v1.8b,v5.8b\n\t"\
    ""IMULL" v24.8h,v1.8b,v6.8b; prfm pldl1keep,[%[a_ptr2],%w[pref],SXTW #0]\n\t"\
    ""IMULL" v25.8h,v2.8b,v4.8b; "IMULL" v26.8h,v2.8b,v5.8b\n\t"\
    ""IMULL" v27.8h,v2.8b,v6.8b; prfm pldl1keep,[%[a_ptr3],%w[pref],SXTW #0]\n\t"\
    ""IMULL" v28.8h,v3.8b,v4.8b; "IMULL" v29.8h,v3.8b,v5.8b\n\t"\
    ""IMULL" v30.8h,v3.8b,v6.8b; prfm pldl1keep,[%[a_ptr4],%w[pref],SXTW #0]\n\t"\
    ""IADALP" %[q1].4s,v19.8h; "IMULL"2 v19.8h,v0.16b,v4.16b\n\t"\
    ""IADALP" %[q2].4s,v20.8h; "IMULL"2 v20.8h,v0.16b,v5.16b\n\t"\
    ""IADALP" %[q3].4s,v21.8h; "IMULL"2 v21.8h,v0.16b,v6.16b\n\t"\
    ""IADALP" v10.4s,v22.8h; "IMULL"2 v22.8h,v1.16b,v4.16b\n\t"\
    ""IADALP" v11.4s,v23.8h; "IMULL"2 v23.8h,v1.16b,v5.16b\n\t"\
    ""IADALP" v12.4s,v24.8h; "IMULL"2 v24.8h,v1.16b,v6.16b\n\t"\
    ""IADALP" v13.4s,v25.8h; "IMULL"2 v25.8h,v2.16b,v4.16b\n\t"\
    ""IADALP" v14.4s,v26.8h; "IMULL"2 v26.8h,v2.16b,v5.16b\n\t"\
    ""IADALP" v15.4s,v27.8h; "IMULL"2 v27.8h,v2.16b,v6.16b\n\t"\
    ""IADALP" v16.4s,v28.8h; "IMULL"2 v28.8h,v3.16b,v4.16b\n\t"\
    ""IADALP" v17.4s,v29.8h; "IMULL"2 v29.8h,v3.16b,v5.16b\n\t"\
    ""IADALP" v18.4s,v30.8h; "IMULL"2 v30.8h,v3.16b,v6.16b\n\t"\
    ""IADALP" %[q1].4s,v19.8h; "IADALP" %[q2].4s,v20.8h; "IADALP" %[q3].4s,v21.8h\n\t"\
    "sub %w[k_left],%w[k_left],#16\n\t"\
    ""IADALP" v10.4s,v22.8h; "IADALP" v11.4s,v23.8h; "IADALP" v12.4s,v24.8h\n\t"\
    ""IADALP" v13.4s,v25.8h; "IADALP" v14.4s,v26.8h; "IADALP" v15.4s,v27.8h\n\t"\
    ""IADALP" v16.4s,v28.8h; "IADALP" v17.4s,v29.8h; "IADALP" v18.4s,v30.8h\n\t"\
    "3:\n\t"\
    "cmp %w[k_left],#8; b.lt 4f\n\t"\
    "ldr d0,[%[a_ptr1]],#8; ldr d1,[%[a_ptr2]],#8\n\t"\
    "ldr d2,[%[a_ptr3]],#8; ldr d3,[%[a_ptr4]],#8\n\t"\
    "ldr d4,[%[b_ptr]]; ldr d5,[%[b_ptr],#8]\n\t"\
    "ldr d6,[%[b_ptr],#16]; add %[b_ptr],%[b_ptr],#24\n\t"\
    "sub %w[k_left],%w[k_left],#8\n\t"\
    ""IMULL" v19.8h,v0.8b,v4.8b; "IMULL" v20.8h,v0.8b,v5.8b\n\t"\
    ""IMULL" v21.8h,v0.8b,v6.8b; "IMULL" v22.8h,v1.8b,v4.8b\n\t"\
    ""IMULL" v23.8h,v1.8b,v5.8b; "IMULL" v24.8h,v1.8b,v6.8b\n\t"\
    ""IMULL" v25.8h,v2.8b,v4.8b; "IMULL" v26.8h,v2.8b,v5.8b\n\t"\
    ""IMULL" v27.8h,v2.8b,v6.8b; "IMULL" v28.8h,v3.8b,v4.8b\n\t"\
    ""IMULL" v29.8h,v3.8b,v5.8b; "IMULL" v30.8h,v3.8b,v6.8b\n\t"\
    ""IADALP" %[q1].4s,v19.8h; "IADALP" %[q2].4s,v20.8h; "IADALP" %[q3].4s,v21.8h\n\t"\
    ""IADALP" v10.4s,v22.8h; "IADALP" v11.4s,v23.8h; "IADALP" v12.4s,v24.8h\n\t"\
    ""IADALP" v13.4s,v25.8h; "IADALP" v14.4s,v26.8h; "IADALP" v15.4s,v27.8h\n\t"\
    ""IADALP" v16.4s,v28.8h; "IADALP" v17.4s,v29.8h; "IADALP" v18.4s,v30.8h\n\t"\
    "4:\n\t"\
    "addp %[q1].4s,%[q1].4s,v10.4s; addp v13.4s,v13.4s,v16.4s\n\t"\
    "addp %[q2].4s,%[q2].4s,v11.4s; addp v14.4s,v14.4s,v17.4s\n\t"\
    "addp %[q3].4s,%[q3].4s,v12.4s; addp v15.4s,v15.4s,v18.4s\n\t"\
    "cmp %w[k_left],#4; b.lt 5f\n\t"\
    "ldr s0,[%[a_ptr1]],#4; ldr s1,[%[a_ptr2]],#4\n\t"\
    "ldr s2,[%[a_ptr3]],#4; ldr s3,[%[a_ptr4]],#4\n\t"\
    "ld1r {v4.2s},[%[b_ptr]],#4; ins v0.s[1],v1.s[0]\n\t"\
    "ld1r {v5.2s},[%[b_ptr]],#4; ins v2.s[1],v3.s[0]\n\t"\
    "ld1r {v6.2s},[%[b_ptr]],#4\n\t"\
    "sub %w[k_left],%w[k_left],#4\n\t"\
    ""IMULL" v19.8h,v0.8b,v4.8b; "IMULL" v20.8h,v0.8b,v5.8b\n\t"\
    ""IMULL" v21.8h,v0.8b,v6.8b; "IMULL" v25.8h,v2.8b,v4.8b\n\t"\
    ""IMULL" v26.8h,v2.8b,v5.8b; "IMULL" v27.8h,v2.8b,v6.8b\n\t"\
    ""IADALP" %[q1].4s,v19.8h; "IADALP" %[q2].4s,v20.8h; "IADALP" %[q3].4s,v21.8h\n\t"\
    ""IADALP" v13.4s,v25.8h; "IADALP" v14.4s,v26.8h; "IADALP" v15.4s,v27.8h\n\t"\
    "5:\n\t"\
    "addp %[q1].4s,%[q1].4s,v13.4s\n\t"\
    "addp %[q2].4s,%[q2].4s,v14.4s\n\t"\
    "addp %[q3].4s,%[q3].4s,v15.4s\n\t"\
    "cmp %w[k_left],#2; b.lt 6f\n\t"\
    "ldr h0,[%[a_ptr1]],#2; ldr h1,[%[a_ptr2]],#2\n\t"\
    "ldr h2,[%[a_ptr3]],#2; ldr h3,[%[a_ptr4]],#2\n\t"\
    "ld1r {v4.4h},[%[b_ptr]],#2; ins v0.h[1],v1.h[0]\n\t"\
    "ld1r {v5.4h},[%[b_ptr]],#2; ins v2.h[1],v3.h[0]\n\t"\
    "ld1r {v6.4h},[%[b_ptr]],#2\n\t"\
    "sub %w[k_left],%w[k_left],#2\n\t"\
    "ins v0.s[1],v2.s[0]\n\t"\
    ""IMULL" v19.8h,v0.8b,v4.8b\n\t"\
    ""IMULL" v20.8h,v0.8b,v5.8b\n\t"\
    ""IMULL" v21.8h,v0.8b,v6.8b\n\t"\
    ""IADALP" %[q1].4s,v19.8h; "IADALP" %[q2].4s,v20.8h; "IADALP" %[q3].4s,v21.8h\n\t"\
    "6:\n\t"\
    "cmp %w[k_left],#1; b.lt 7f\n\t"\
    "ldr b0,[%[a_ptr1]],#1; ldr b1,[%[a_ptr2]],#1\n\t"\
    "ldr b2,[%[a_ptr3]],#1; ldr b3,[%[a_ptr4]],#1\n\t"\
    "ldr b4,[%[b_ptr]]; ins v0.b[1],v1.b[0]\n\t"\
    "ldr b5,[%[b_ptr],#1]; ins v2.b[1],v3.b[0]\n\t"\
    "ldr b6,[%[b_ptr],#2]; add %[b_ptr],%[b_ptr],#3\n\t"\
    "ins v4.b[1],v5.b[0]\n\t"\
    "ins v0.h[1],v2.h[0]; ins v4.b[2],v6.b[0]\n\t"\
    "sub %w[k_left],%w[k_left],#1\n\t"\
    ""IXTL" v0.8h,v0.8b; "IXTL" v4.8h,v4.8b\n\t"\
    ""IMLAL" %[q1].4s,v0.4h,v4.h[0]\n\t"\
    ""IMLAL" %[q2].4s,v0.4h,v4.h[1]\n\t"\
    ""IMLAL" %[q3].4s,v0.4h,v4.h[2]\n\t"\
    "7:\n\t"\
   :[q1]"=w"(cq1), [q2]"=w"(cq2), [q3]"=w"(cq3), [k_left]"+r"(k_left),\
    [a_ptr1]"+r"(a_ptr1), [a_ptr2]"+r"(a_ptr2), [a_ptr3]"+r"(a_ptr3),\
    [a_ptr4]"+r"(a_ptr4), [b_ptr]"+r"(b_ptr)\
   :[pref]"r"(next_pref)\
   :"cc","memory","v0","v1","v2","v3","v4","v5","v6",\
    "v10","v11","v12","v13","v14","v15","v16","v17","v18",\
    "v19","v20","v21","v22","v23","v24","v25","v26","v27",\
    "v28","v29","v30");\
\
  if (c_rowmajor) {\
    I32X4X3 cqt1 = VLD3Q_I32(c_ptr);\
    cqt1.val[0] = VMLAQ_N_I32(cq1, cqt1.val[0], beta);\
    cqt1.val[1] = VMLAQ_N_I32(cq2, cqt1.val[1], beta);\
    cqt1.val[2] = VMLAQ_N_I32(cq3, cqt1.val[2], beta);\
    VST3Q_I32(c_ptr, cqt1);\
  } else {\
    cq1 = VMLAQ_N_I32(cq1, VLD1Q_I32(c_ptr), beta);\
    cq2 = VMLAQ_N_I32(cq2, VLD1Q_I32(c_ptr + LDM), beta);\
    cq3 = VMLAQ_N_I32(cq3, VLD1Q_I32(c_ptr + LDM * 2), beta);\
    VST1Q_I32(c_ptr, cq1); c_ptr += LDM;\
    VST1Q_I32(c_ptr, cq2); c_ptr += LDM;\
    VST1Q_I32(c_ptr, cq3);\
  }\
}



#define I8I32MLA_SKINNY_DOT_INLINE_FUNCS_M4(gemm) \
  I8I32MLA_SKINNYDOT_INLINE_M4N1(gemm)\
  I8I32MLA_SKINNYDOT_INLINE_M4N2(gemm)\
  I8I32MLA_SKINNYDOT_INLINE_M4N3(gemm)

I8I32MLA_SKINNY_DOT_INLINE_FUNCS_M4(I8I32MLAGEMM)

#define GEMM_SKINNY_DOT_INLINE_FUNC_DEDUCE(a, b, c, d)\
  GEMM_SKINNY_DOT_INLINE_PACK_FUNC(a, b, c, d)

GEMM_SKINNY_DOT_INLINE_FUNC_DEDUCE(I8I32MLAGEMM, 1, 1, 31)
GEMM_SKINNY_DOT_INLINE_FUNC_DEDUCE(I8I32MLAGEMM, 1, 2, 31)
GEMM_SKINNY_DOT_INLINE_FUNC_DEDUCE(I8I32MLAGEMM, 1, 3, 31)

static inline bool unroll_test_m4n1(uint32_t M, uint32_t K) {
  return K <= 16384;
}

static inline bool unroll_test_m1n1(uint32_t M, uint32_t K) {
  return true;
}

static inline bool unroll_test_m4n2(uint32_t M, uint32_t K) {
  return K <= 16384;
}

static inline bool unroll_test_m1n2(uint32_t M, uint32_t K) {
  return true;
}

static inline bool unroll_test_m4n3(uint32_t M, uint32_t K) {
  return K <= 16384;
}

static inline bool unroll_test_m1n3(uint32_t M, uint32_t K) {
  return true;
}

#endif