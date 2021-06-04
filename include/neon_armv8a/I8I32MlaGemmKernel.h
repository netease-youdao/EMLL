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


#include "arm_neon/NeonI8I32MlaGemmKernel.h"

#ifndef INCLUDE_ARMV8A_I8I32MLA_ASM_KERNEL
#define INCLUDE_ARMV8A_I8I32MLA_ASM_KERNEL

static inline void pref_c_8(const I32 *c) {
  __asm__("prfm pstl1keep,[%0]; prfm pstl1keep,[%0,#32]\n\t"::"r"(c):);
}

static inline void pref_c_12(const I32 *c) {
  __asm__("prfm pstl1keep,[%0]; prfm pstl1keep,[%0,#48]\n\t"::"r"(c):);
}

#define KERNEL_M8N12 \
  const I32 *c_pref = c_ptr;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref); c_pref += ldc;\
  pref_c_8(c_pref);\
  COMMON_KERNEL_HEADER(a_head, b_head)\
  I32X4 cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  I32X4 cq09, cq10, cq11, cq12, cq13, cq14, cq15, cq16;\
  I32X4 cq17, cq18, cq19, cq20, cq21, cq22, cq23, cq24;\
  __asm__ __volatile__(\
    "movi %[cq01].16b,#0; movi %[cq02].16b,#0\n\t"\
    "movi %[cq03].16b,#0; movi %[cq04].16b,#0\n\t"\
    "movi %[cq05].16b,#0; movi %[cq06].16b,#0\n\t"\
    "movi %[cq07].16b,#0; movi %[cq08].16b,#0\n\t"\
    "movi %[cq09].16b,#0; movi %[cq10].16b,#0\n\t"\
    "movi %[cq11].16b,#0; movi %[cq12].16b,#0\n\t"\
    "movi %[cq13].16b,#0; movi %[cq14].16b,#0\n\t"\
    "movi %[cq15].16b,#0; movi %[cq16].16b,#0\n\t"\
    "movi %[cq17].16b,#0; movi %[cq18].16b,#0\n\t"\
    "movi %[cq19].16b,#0; movi %[cq20].16b,#0\n\t"\
    "movi %[cq21].16b,#0; movi %[cq22].16b,#0\n\t"\
    "movi %[cq23].16b,#0; movi %[cq24].16b,#0\n\t"\
    "cmp %w[k_left],#1; b.lt 4f\n\t"\
    "ldr q0,[%[a_ptr]],#16\n\t"\
    "ldr q2,[%[b_ptr]]; ldr d3,[%[b_ptr],#16]; add %[b_ptr],%[b_ptr],#24\n\t"\
    "cmp %w[k_left],#3; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    ""IMLAL" %[cq01].4s,v0.4h,v2.h[0]; ldr x0,[%[b_ptr]],#48\n\t"\
    ""IMLAL2" %[cq02].4s,v0.8h,v2.h[0]\n\t"\
    ""IMLAL" %[cq03].4s,v0.4h,v2.h[1]\n\t"\
    ""IMLAL2" %[cq04].4s,v0.8h,v2.h[1]\n\t"\
    ""IMLAL" %[cq05].4s,v0.4h,v2.h[2]\n\t"\
    ""IMLAL2" %[cq06].4s,v0.8h,v2.h[2]\n\t"\
    ""IMLAL" %[cq07].4s,v0.4h,v2.h[3]\n\t"\
    ""IMLAL2" %[cq08].4s,v0.8h,v2.h[3]\n\t"\
    "fmov v3.d[1],x0; ldr d1,[%[a_ptr]],#32\n\t"\
    ""IMLAL" %[cq09].4s,v0.4h,v2.h[4]\n\t"\
    ""IMLAL2" %[cq10].4s,v0.8h,v2.h[4]; ldr x0,[%[a_ptr],#-24]\n\t"\
    ""IMLAL" %[cq11].4s,v0.4h,v2.h[5]\n\t"\
    ""IMLAL2" %[cq12].4s,v0.8h,v2.h[5]\n\t"\
    ""IMLAL" %[cq13].4s,v0.4h,v2.h[6]\n\t"\
    ""IMLAL2" %[cq14].4s,v0.8h,v2.h[6]\n\t"\
    ""IMLAL" %[cq15].4s,v0.4h,v2.h[7]\n\t"\
    ""IMLAL2" %[cq16].4s,v0.8h,v2.h[7]\n\t"\
    "fmov v1.d[1],x0; ldr d4,[%[b_ptr],#-40]\n\t"\
    ""IMLAL" %[cq17].4s,v0.4h,v3.h[0]; ldr x0,[%[b_ptr],#-32]\n\t"\
    ""IMLAL2" %[cq18].4s,v0.8h,v3.h[0]\n\t"\
    ""IMLAL" %[cq19].4s,v0.4h,v3.h[1]\n\t"\
    ""IMLAL2" %[cq20].4s,v0.8h,v3.h[1]\n\t"\
    ""IMLAL" %[cq21].4s,v0.4h,v3.h[2]\n\t"\
    ""IMLAL2" %[cq22].4s,v0.8h,v3.h[2]\n\t"\
    ""IMLAL" %[cq23].4s,v0.4h,v3.h[3]\n\t"\
    ""IMLAL2" %[cq24].4s,v0.8h,v3.h[3]\n\t"\
    "fmov v4.d[1],x0; ldr d0,[%[a_ptr],#-16]\n\t"\
    ""IMLAL" %[cq01].4s,v1.4h,v3.h[4]; ldr x0,[%[a_ptr],#-8]\n\t"\
    ""IMLAL2" %[cq02].4s,v1.8h,v3.h[4]\n\t"\
    ""IMLAL" %[cq03].4s,v1.4h,v3.h[5]\n\t"\
    ""IMLAL2" %[cq04].4s,v1.8h,v3.h[5]\n\t"\
    ""IMLAL" %[cq05].4s,v1.4h,v3.h[6]\n\t"\
    ""IMLAL2" %[cq06].4s,v1.8h,v3.h[6]\n\t"\
    ""IMLAL" %[cq07].4s,v1.4h,v3.h[7]\n\t"\
    ""IMLAL2" %[cq08].4s,v1.8h,v3.h[7]\n\t"\
    "fmov v0.d[1],x0; ldr d2,[%[b_ptr],#-24]\n\t"\
    ""IMLAL" %[cq09].4s,v1.4h,v4.h[0]; ldr x0,[%[b_ptr],#-16]\n\t"\
    ""IMLAL2" %[cq10].4s,v1.8h,v4.h[0]\n\t"\
    ""IMLAL" %[cq11].4s,v1.4h,v4.h[1]\n\t"\
    ""IMLAL2" %[cq12].4s,v1.8h,v4.h[1]\n\t"\
    ""IMLAL" %[cq13].4s,v1.4h,v4.h[2]\n\t"\
    ""IMLAL2" %[cq14].4s,v1.8h,v4.h[2]\n\t"\
    ""IMLAL" %[cq15].4s,v1.4h,v4.h[3]\n\t"\
    ""IMLAL2" %[cq16].4s,v1.8h,v4.h[3]\n\t"\
    "fmov v2.d[1],x0; ldr d3,[%[b_ptr],#-8]\n\t"\
    ""IMLAL" %[cq17].4s,v1.4h,v4.h[4]\n\t"\
    ""IMLAL2" %[cq18].4s,v1.8h,v4.h[4]\n\t"\
    ""IMLAL" %[cq19].4s,v1.4h,v4.h[5]\n\t"\
    ""IMLAL2" %[cq20].4s,v1.8h,v4.h[5]; sub %w[k_left],%w[k_left],#2\n\t"\
    ""IMLAL" %[cq21].4s,v1.4h,v4.h[6]\n\t"\
    ""IMLAL2" %[cq22].4s,v1.8h,v4.h[6]; cmp %w[k_left],#3\n\t"\
    ""IMLAL" %[cq23].4s,v1.4h,v4.h[7]\n\t"\
    ""IMLAL2" %[cq24].4s,v1.8h,v4.h[7]; b.ge 1b\n\t"\
    "2:\n\t"\
    "cmp %w[k_left],#2; b.lt 3f\n\t"\
    ""IMLAL" %[cq01].4s,v0.4h,v2.h[0]; ldr x0,[%[b_ptr]],#24\n\t"\
    ""IMLAL2" %[cq02].4s,v0.8h,v2.h[0]\n\t"\
    ""IMLAL" %[cq03].4s,v0.4h,v2.h[1]\n\t"\
    ""IMLAL2" %[cq04].4s,v0.8h,v2.h[1]\n\t"\
    ""IMLAL" %[cq05].4s,v0.4h,v2.h[2]\n\t"\
    ""IMLAL2" %[cq06].4s,v0.8h,v2.h[2]\n\t"\
    ""IMLAL" %[cq07].4s,v0.4h,v2.h[3]\n\t"\
    ""IMLAL2" %[cq08].4s,v0.8h,v2.h[3]\n\t"\
    "fmov v3.d[1],x0; ldr d1,[%[a_ptr]],#16\n\t"\
    ""IMLAL" %[cq09].4s,v0.4h,v2.h[4]\n\t"\
    ""IMLAL2" %[cq10].4s,v0.8h,v2.h[4]; ldr x0,[%[a_ptr],#-8]\n\t"\
    ""IMLAL" %[cq11].4s,v0.4h,v2.h[5]\n\t"\
    ""IMLAL2" %[cq12].4s,v0.8h,v2.h[5]\n\t"\
    ""IMLAL" %[cq13].4s,v0.4h,v2.h[6]\n\t"\
    ""IMLAL2" %[cq14].4s,v0.8h,v2.h[6]\n\t"\
    ""IMLAL" %[cq15].4s,v0.4h,v2.h[7]\n\t"\
    ""IMLAL2" %[cq16].4s,v0.8h,v2.h[7]\n\t"\
    "fmov v1.d[1],x0; ldr d4,[%[b_ptr],#-16]\n\t"\
    ""IMLAL" %[cq17].4s,v0.4h,v3.h[0]; ldr x0,[%[b_ptr],#-8]\n\t"\
    ""IMLAL2" %[cq18].4s,v0.8h,v3.h[0]\n\t"\
    ""IMLAL" %[cq19].4s,v0.4h,v3.h[1]\n\t"\
    ""IMLAL2" %[cq20].4s,v0.8h,v3.h[1]\n\t"\
    ""IMLAL" %[cq21].4s,v0.4h,v3.h[2]\n\t"\
    ""IMLAL2" %[cq22].4s,v0.8h,v3.h[2]\n\t"\
    ""IMLAL" %[cq23].4s,v0.4h,v3.h[3]\n\t"\
    ""IMLAL2" %[cq24].4s,v0.8h,v3.h[3]\n\t"\
    "fmov v4.d[1],x0\n\t"\
    ""IMLAL" %[cq01].4s,v1.4h,v3.h[4]\n\t"\
    ""IMLAL2" %[cq02].4s,v1.8h,v3.h[4]\n\t"\
    ""IMLAL" %[cq03].4s,v1.4h,v3.h[5]\n\t"\
    ""IMLAL2" %[cq04].4s,v1.8h,v3.h[5]\n\t"\
    ""IMLAL" %[cq05].4s,v1.4h,v3.h[6]\n\t"\
    ""IMLAL2" %[cq06].4s,v1.8h,v3.h[6]\n\t"\
    ""IMLAL" %[cq07].4s,v1.4h,v3.h[7]\n\t"\
    ""IMLAL2" %[cq08].4s,v1.8h,v3.h[7]\n\t"\
    ""IMLAL" %[cq09].4s,v1.4h,v4.h[0]\n\t"\
    ""IMLAL2" %[cq10].4s,v1.8h,v4.h[0]\n\t"\
    ""IMLAL" %[cq11].4s,v1.4h,v4.h[1]\n\t"\
    ""IMLAL2" %[cq12].4s,v1.8h,v4.h[1]\n\t"\
    ""IMLAL" %[cq13].4s,v1.4h,v4.h[2]\n\t"\
    ""IMLAL2" %[cq14].4s,v1.8h,v4.h[2]\n\t"\
    ""IMLAL" %[cq15].4s,v1.4h,v4.h[3]\n\t"\
    ""IMLAL2" %[cq16].4s,v1.8h,v4.h[3]\n\t"\
    ""IMLAL" %[cq17].4s,v1.4h,v4.h[4]\n\t"\
    ""IMLAL2" %[cq18].4s,v1.8h,v4.h[4]\n\t"\
    ""IMLAL" %[cq19].4s,v1.4h,v4.h[5]\n\t"\
    ""IMLAL2" %[cq20].4s,v1.8h,v4.h[5]; sub %w[k_left],%w[k_left],#2\n\t"\
    ""IMLAL" %[cq21].4s,v1.4h,v4.h[6]\n\t"\
    ""IMLAL2" %[cq22].4s,v1.8h,v4.h[6]\n\t"\
    ""IMLAL" %[cq23].4s,v1.4h,v4.h[7]\n\t"\
    ""IMLAL2" %[cq24].4s,v1.8h,v4.h[7]; b 4f\n\t"\
    "3:\n\t"\
    ""IMLAL" %[cq01].4s,v0.4h,v2.h[0]; "IMLAL2" %[cq02].4s,v0.8h,v2.h[0]\n\t"\
    ""IMLAL" %[cq03].4s,v0.4h,v2.h[1]; "IMLAL2" %[cq04].4s,v0.8h,v2.h[1]\n\t"\
    ""IMLAL" %[cq05].4s,v0.4h,v2.h[2]; "IMLAL2" %[cq06].4s,v0.8h,v2.h[2]\n\t"\
    ""IMLAL" %[cq07].4s,v0.4h,v2.h[3]; "IMLAL2" %[cq08].4s,v0.8h,v2.h[3]\n\t"\
    ""IMLAL" %[cq09].4s,v0.4h,v2.h[4]; "IMLAL2" %[cq10].4s,v0.8h,v2.h[4]\n\t"\
    ""IMLAL" %[cq11].4s,v0.4h,v2.h[5]; "IMLAL2" %[cq12].4s,v0.8h,v2.h[5]\n\t"\
    ""IMLAL" %[cq13].4s,v0.4h,v2.h[6]; "IMLAL2" %[cq14].4s,v0.8h,v2.h[6]\n\t"\
    ""IMLAL" %[cq15].4s,v0.4h,v2.h[7]; "IMLAL2" %[cq16].4s,v0.8h,v2.h[7]\n\t"\
    ""IMLAL" %[cq17].4s,v0.4h,v3.h[0]; "IMLAL2" %[cq18].4s,v0.8h,v3.h[0]\n\t"\
    ""IMLAL" %[cq19].4s,v0.4h,v3.h[1]; "IMLAL2" %[cq20].4s,v0.8h,v3.h[1]\n\t"\
    ""IMLAL" %[cq21].4s,v0.4h,v3.h[2]; "IMLAL2" %[cq22].4s,v0.8h,v3.h[2]\n\t"\
    ""IMLAL" %[cq23].4s,v0.4h,v3.h[3]; "IMLAL2" %[cq24].4s,v0.8h,v3.h[3]\n\t"\
    "sub %w[k_left],%w[k_left],#1\n\t"\
    "4:\n\t"\
   :[a_ptr]"+r"(a_ptr), [b_ptr]"+r"(b_ptr), [k_left]"+r"(k_left),\
   [cq01]"=w"(cq01), [cq02]"=w"(cq02), [cq03]"=w"(cq03), [cq04]"=w"(cq04),\
   [cq05]"=w"(cq05), [cq06]"=w"(cq06), [cq07]"=w"(cq07), [cq08]"=w"(cq08),\
   [cq09]"=w"(cq09), [cq10]"=w"(cq10), [cq11]"=w"(cq11), [cq12]"=w"(cq12),\
   [cq13]"=w"(cq13), [cq14]"=w"(cq14), [cq15]"=w"(cq15), [cq16]"=w"(cq16),\
   [cq17]"=w"(cq17), [cq18]"=w"(cq18), [cq19]"=w"(cq19), [cq20]"=w"(cq20),\
   [cq21]"=w"(cq21), [cq22]"=w"(cq22), [cq23]"=w"(cq23), [cq24]"=w"(cq24)\
   ::"cc","memory","x0","v0","v1","v2","v3","v4");

#define SAVE_M8N12 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M8N2(cq01, cq02, cq03, cq04)\
  UNIT_SAVE_M8N2(cq05, cq06, cq07, cq08)\
  UNIT_SAVE_M8N2(cq09, cq10, cq11, cq12)\
  UNIT_SAVE_M8N2(cq13, cq14, cq15, cq16)\
  UNIT_SAVE_M8N2(cq17, cq18, cq19, cq20)\
  UNIT_SAVE_M8N2(cq21, cq22, cq23, cq24)

#define KERNEL_M12N8 \
  const I32 *c_pref = c_ptr;\
  pref_c_12(c_pref); c_pref += ldc;\
  pref_c_12(c_pref); c_pref += ldc;\
  pref_c_12(c_pref); c_pref += ldc;\
  pref_c_12(c_pref); c_pref += ldc;\
  pref_c_12(c_pref); c_pref += ldc;\
  pref_c_12(c_pref); c_pref += ldc;\
  pref_c_12(c_pref); c_pref += ldc;\
  pref_c_12(c_pref);\
  COMMON_KERNEL_HEADER(a_head, b_head)\
  I32X4 cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  I32X4 cq09, cq10, cq11, cq12, cq13, cq14, cq15, cq16;\
  I32X4 cq17, cq18, cq19, cq20, cq21, cq22, cq23, cq24;\
  __asm__ __volatile__(\
    "movi %[cq01].16b,#0; movi %[cq02].16b,#0\n\t"\
    "movi %[cq03].16b,#0; movi %[cq04].16b,#0\n\t"\
    "movi %[cq05].16b,#0; movi %[cq06].16b,#0\n\t"\
    "movi %[cq07].16b,#0; movi %[cq08].16b,#0\n\t"\
    "movi %[cq09].16b,#0; movi %[cq10].16b,#0\n\t"\
    "movi %[cq11].16b,#0; movi %[cq12].16b,#0\n\t"\
    "movi %[cq13].16b,#0; movi %[cq14].16b,#0\n\t"\
    "movi %[cq15].16b,#0; movi %[cq16].16b,#0\n\t"\
    "movi %[cq17].16b,#0; movi %[cq18].16b,#0\n\t"\
    "movi %[cq19].16b,#0; movi %[cq20].16b,#0\n\t"\
    "movi %[cq21].16b,#0; movi %[cq22].16b,#0\n\t"\
    "movi %[cq23].16b,#0; movi %[cq24].16b,#0\n\t"\
    "cmp %w[k_left],#1; b.lt 4f\n\t"\
    "ldr q0,[%[a_ptr]]; ldr d1,[%[a_ptr],#16]; add %[a_ptr],%[a_ptr],#24\n\t"\
    "ldr q3,[%[b_ptr]],#16\n\t"\
    "cmp %w[k_left],#3; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    ""IMLAL" %[cq01].4s,v0.4h,v3.h[0]; ldr x0,[%[a_ptr]],#48\n\t"\
    ""IMLAL" %[cq04].4s,v0.4h,v3.h[1]\n\t"\
    ""IMLAL" %[cq07].4s,v0.4h,v3.h[2]\n\t"\
    ""IMLAL" %[cq10].4s,v0.4h,v3.h[3]\n\t"\
    ""IMLAL" %[cq13].4s,v0.4h,v3.h[4]\n\t"\
    ""IMLAL" %[cq16].4s,v0.4h,v3.h[5]\n\t"\
    ""IMLAL" %[cq19].4s,v0.4h,v3.h[6]\n\t"\
    ""IMLAL" %[cq22].4s,v0.4h,v3.h[7]\n\t"\
    "fmov v1.d[1],x0; ldr d4,[%[b_ptr]],#32\n\t"\
    ""IMLAL2" %[cq02].4s,v0.8h,v3.h[0]\n\t"\
    ""IMLAL2" %[cq05].4s,v0.8h,v3.h[1]; ldr x0,[%[b_ptr],#-24]\n\t"\
    ""IMLAL2" %[cq08].4s,v0.8h,v3.h[2]\n\t"\
    ""IMLAL2" %[cq11].4s,v0.8h,v3.h[3]\n\t"\
    ""IMLAL2" %[cq14].4s,v0.8h,v3.h[4]\n\t"\
    ""IMLAL2" %[cq17].4s,v0.8h,v3.h[5]\n\t"\
    ""IMLAL2" %[cq20].4s,v0.8h,v3.h[6]\n\t"\
    ""IMLAL2" %[cq23].4s,v0.8h,v3.h[7]\n\t"\
    "fmov v4.d[1],x0; ldr d2,[%[a_ptr],#-40]\n\t"\
    ""IMLAL" %[cq03].4s,v1.4h,v3.h[0]; ldr x0,[%[a_ptr],#-32]\n\t"\
    ""IMLAL" %[cq06].4s,v1.4h,v3.h[1]\n\t"\
    ""IMLAL" %[cq09].4s,v1.4h,v3.h[2]\n\t"\
    ""IMLAL" %[cq12].4s,v1.4h,v3.h[3]\n\t"\
    ""IMLAL" %[cq15].4s,v1.4h,v3.h[4]\n\t"\
    ""IMLAL" %[cq18].4s,v1.4h,v3.h[5]\n\t"\
    ""IMLAL" %[cq21].4s,v1.4h,v3.h[6]\n\t"\
    ""IMLAL" %[cq24].4s,v1.4h,v3.h[7]\n\t"\
    "fmov v2.d[1],x0; ldr d3,[%[b_ptr],#-16]\n\t"\
    ""IMLAL2" %[cq01].4s,v1.8h,v4.h[0]; ldr x0,[%[b_ptr],#-8]\n\t"\
    ""IMLAL2" %[cq04].4s,v1.8h,v4.h[1]\n\t"\
    ""IMLAL2" %[cq07].4s,v1.8h,v4.h[2]\n\t"\
    ""IMLAL2" %[cq10].4s,v1.8h,v4.h[3]\n\t"\
    ""IMLAL2" %[cq13].4s,v1.8h,v4.h[4]\n\t"\
    ""IMLAL2" %[cq16].4s,v1.8h,v4.h[5]\n\t"\
    ""IMLAL2" %[cq19].4s,v1.8h,v4.h[6]\n\t"\
    ""IMLAL2" %[cq22].4s,v1.8h,v4.h[7]\n\t"\
    "fmov v3.d[1],x0; ldr d0,[%[a_ptr],#-24]\n\t"\
    ""IMLAL" %[cq02].4s,v2.4h,v4.h[0]; ldr x0,[%[a_ptr],#-16]\n\t"\
    ""IMLAL" %[cq05].4s,v2.4h,v4.h[1]\n\t"\
    ""IMLAL" %[cq08].4s,v2.4h,v4.h[2]\n\t"\
    ""IMLAL" %[cq11].4s,v2.4h,v4.h[3]\n\t"\
    ""IMLAL" %[cq14].4s,v2.4h,v4.h[4]\n\t"\
    ""IMLAL" %[cq17].4s,v2.4h,v4.h[5]\n\t"\
    ""IMLAL" %[cq20].4s,v2.4h,v4.h[6]\n\t"\
    ""IMLAL" %[cq23].4s,v2.4h,v4.h[7]\n\t"\
    "fmov v0.d[1],x0; ldr d1,[%[a_ptr],#-8]\n\t"\
    ""IMLAL2" %[cq03].4s,v2.8h,v4.h[0]\n\t"\
    ""IMLAL2" %[cq06].4s,v2.8h,v4.h[1]\n\t"\
    ""IMLAL2" %[cq09].4s,v2.8h,v4.h[2]\n\t"\
    ""IMLAL2" %[cq12].4s,v2.8h,v4.h[3]; sub %w[k_left],%w[k_left],#2\n\t"\
    ""IMLAL2" %[cq15].4s,v2.8h,v4.h[4]\n\t"\
    ""IMLAL2" %[cq18].4s,v2.8h,v4.h[5]; cmp %w[k_left],#3\n\t"\
    ""IMLAL2" %[cq21].4s,v2.8h,v4.h[6]\n\t"\
    ""IMLAL2" %[cq24].4s,v2.8h,v4.h[7]; b.ge 1b\n\t"\
    "2:\n\t"\
    "cmp %w[k_left],#2; b.lt 3f\n\t"\
    ""IMLAL" %[cq01].4s,v0.4h,v3.h[0]; ldr x0,[%[a_ptr]],#24\n\t"\
    ""IMLAL" %[cq04].4s,v0.4h,v3.h[1]\n\t"\
    ""IMLAL" %[cq07].4s,v0.4h,v3.h[2]\n\t"\
    ""IMLAL" %[cq10].4s,v0.4h,v3.h[3]\n\t"\
    ""IMLAL" %[cq13].4s,v0.4h,v3.h[4]\n\t"\
    ""IMLAL" %[cq16].4s,v0.4h,v3.h[5]\n\t"\
    ""IMLAL" %[cq19].4s,v0.4h,v3.h[6]\n\t"\
    ""IMLAL" %[cq22].4s,v0.4h,v3.h[7]\n\t"\
    "fmov v1.d[1],x0; ldr d4,[%[b_ptr]],#16\n\t"\
    ""IMLAL2" %[cq02].4s,v0.8h,v3.h[0]\n\t"\
    ""IMLAL2" %[cq05].4s,v0.8h,v3.h[1]; ldr x0,[%[b_ptr],#-8]\n\t"\
    ""IMLAL2" %[cq08].4s,v0.8h,v3.h[2]\n\t"\
    ""IMLAL2" %[cq11].4s,v0.8h,v3.h[3]\n\t"\
    ""IMLAL2" %[cq14].4s,v0.8h,v3.h[4]\n\t"\
    ""IMLAL2" %[cq17].4s,v0.8h,v3.h[5]\n\t"\
    ""IMLAL2" %[cq20].4s,v0.8h,v3.h[6]\n\t"\
    ""IMLAL2" %[cq23].4s,v0.8h,v3.h[7]\n\t"\
    "fmov v4.d[1],x0; ldr d2,[%[a_ptr],#-16]\n\t"\
    ""IMLAL" %[cq03].4s,v1.4h,v3.h[0]; ldr x0,[%[a_ptr],#-8]\n\t"\
    ""IMLAL" %[cq06].4s,v1.4h,v3.h[1]\n\t"\
    ""IMLAL" %[cq09].4s,v1.4h,v3.h[2]\n\t"\
    ""IMLAL" %[cq12].4s,v1.4h,v3.h[3]\n\t"\
    ""IMLAL" %[cq15].4s,v1.4h,v3.h[4]\n\t"\
    ""IMLAL" %[cq18].4s,v1.4h,v3.h[5]\n\t"\
    ""IMLAL" %[cq21].4s,v1.4h,v3.h[6]\n\t"\
    ""IMLAL" %[cq24].4s,v1.4h,v3.h[7]\n\t"\
    "fmov v2.d[1],x0\n\t"\
    ""IMLAL2" %[cq01].4s,v1.8h,v4.h[0]\n\t"\
    ""IMLAL2" %[cq04].4s,v1.8h,v4.h[1]\n\t"\
    ""IMLAL2" %[cq07].4s,v1.8h,v4.h[2]\n\t"\
    ""IMLAL2" %[cq10].4s,v1.8h,v4.h[3]\n\t"\
    ""IMLAL2" %[cq13].4s,v1.8h,v4.h[4]\n\t"\
    ""IMLAL2" %[cq16].4s,v1.8h,v4.h[5]\n\t"\
    ""IMLAL2" %[cq19].4s,v1.8h,v4.h[6]\n\t"\
    ""IMLAL2" %[cq22].4s,v1.8h,v4.h[7]\n\t"\
    ""IMLAL" %[cq02].4s,v2.4h,v4.h[0]\n\t"\
    ""IMLAL" %[cq05].4s,v2.4h,v4.h[1]\n\t"\
    ""IMLAL" %[cq08].4s,v2.4h,v4.h[2]\n\t"\
    ""IMLAL" %[cq11].4s,v2.4h,v4.h[3]\n\t"\
    ""IMLAL" %[cq14].4s,v2.4h,v4.h[4]\n\t"\
    ""IMLAL" %[cq17].4s,v2.4h,v4.h[5]\n\t"\
    ""IMLAL" %[cq20].4s,v2.4h,v4.h[6]\n\t"\
    ""IMLAL" %[cq23].4s,v2.4h,v4.h[7]\n\t"\
    ""IMLAL2" %[cq03].4s,v2.8h,v4.h[0]\n\t"\
    ""IMLAL2" %[cq06].4s,v2.8h,v4.h[1]\n\t"\
    ""IMLAL2" %[cq09].4s,v2.8h,v4.h[2]\n\t"\
    ""IMLAL2" %[cq12].4s,v2.8h,v4.h[3]; sub %w[k_left],%w[k_left],#2\n\t"\
    ""IMLAL2" %[cq15].4s,v2.8h,v4.h[4]\n\t"\
    ""IMLAL2" %[cq18].4s,v2.8h,v4.h[5]\n\t"\
    ""IMLAL2" %[cq21].4s,v2.8h,v4.h[6]\n\t"\
    ""IMLAL2" %[cq24].4s,v2.8h,v4.h[7]; b 4f\n\t"\
    "3:\n\t"\
    ""IMLAL" %[cq01].4s,v0.4h,v3.h[0]; "IMLAL" %[cq04].4s,v0.4h,v3.h[1]\n\t"\
    ""IMLAL" %[cq07].4s,v0.4h,v3.h[2]; "IMLAL" %[cq10].4s,v0.4h,v3.h[3]\n\t"\
    ""IMLAL" %[cq13].4s,v0.4h,v3.h[4]; "IMLAL" %[cq16].4s,v0.4h,v3.h[5]\n\t"\
    ""IMLAL" %[cq19].4s,v0.4h,v3.h[6]; "IMLAL" %[cq22].4s,v0.4h,v3.h[7]\n\t"\
    ""IMLAL2" %[cq02].4s,v0.8h,v3.h[0]; "IMLAL2" %[cq05].4s,v0.8h,v3.h[1]\n\t"\
    ""IMLAL2" %[cq08].4s,v0.8h,v3.h[2]; "IMLAL2" %[cq11].4s,v0.8h,v3.h[3]\n\t"\
    ""IMLAL2" %[cq14].4s,v0.8h,v3.h[4]; "IMLAL2" %[cq17].4s,v0.8h,v3.h[5]\n\t"\
    ""IMLAL2" %[cq20].4s,v0.8h,v3.h[6]; "IMLAL2" %[cq23].4s,v0.8h,v3.h[7]\n\t"\
    ""IMLAL" %[cq03].4s,v1.4h,v3.h[0]; "IMLAL" %[cq06].4s,v1.4h,v3.h[1]\n\t"\
    ""IMLAL" %[cq09].4s,v1.4h,v3.h[2]; "IMLAL" %[cq12].4s,v1.4h,v3.h[3]\n\t"\
    ""IMLAL" %[cq15].4s,v1.4h,v3.h[4]; "IMLAL" %[cq18].4s,v1.4h,v3.h[5]\n\t"\
    ""IMLAL" %[cq21].4s,v1.4h,v3.h[6]; "IMLAL" %[cq24].4s,v1.4h,v3.h[7]\n\t"\
    "sub %w[k_left],%w[k_left],#1\n\t"\
    "4:\n\t"\
   :[a_ptr]"+r"(a_ptr), [b_ptr]"+r"(b_ptr), [k_left]"+r"(k_left),\
   [cq01]"=w"(cq01), [cq02]"=w"(cq02), [cq03]"=w"(cq03), [cq04]"=w"(cq04),\
   [cq05]"=w"(cq05), [cq06]"=w"(cq06), [cq07]"=w"(cq07), [cq08]"=w"(cq08),\
   [cq09]"=w"(cq09), [cq10]"=w"(cq10), [cq11]"=w"(cq11), [cq12]"=w"(cq12),\
   [cq13]"=w"(cq13), [cq14]"=w"(cq14), [cq15]"=w"(cq15), [cq16]"=w"(cq16),\
   [cq17]"=w"(cq17), [cq18]"=w"(cq18), [cq19]"=w"(cq19), [cq20]"=w"(cq20),\
   [cq21]"=w"(cq21), [cq22]"=w"(cq22), [cq23]"=w"(cq23), [cq24]"=w"(cq24)\
   ::"cc","memory","x0","v0","v1","v2","v3","v4");

#define SAVE_M12N8 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M12N2(cq01, cq02, cq03, cq04, cq05, cq06)\
  UNIT_SAVE_M12N2(cq07, cq08, cq09, cq10, cq11, cq12)\
  UNIT_SAVE_M12N2(cq13, cq14, cq15, cq16, cq17, cq18)\
  UNIT_SAVE_M12N2(cq19, cq20, cq21, cq22, cq23, cq24)

IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 8, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 12, I16, I32)

#endif
