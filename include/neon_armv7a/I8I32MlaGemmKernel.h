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

#ifndef INCLUDE_ARMV7A_I8I32MLA_KERNEL
#define INCLUDE_ARMV7A_I8I32MLA_KERNEL

#define KERNEL_M6N8_UNIT(a_head, b_head) \
  I32X4 cq01, cq02, cq03, cq04, cq05, cq06;\
  I32X4 cq07, cq08, cq09, cq10, cq11, cq12;\
  COMMON_KERNEL_HEADER(a_head, b_head)\
  __asm__ __volatile__(\
    "vmov.i8 %q[cq01],#0; vmov.i8 %q[cq02],#0\n\t"\
    "vmov.i8 %q[cq03],#0; vmov.i8 %q[cq04],#0\n\t"\
    "vmov.i8 %q[cq05],#0; vmov.i8 %q[cq06],#0\n\t"\
    "vmov.i8 %q[cq07],#0; vmov.i8 %q[cq08],#0\n\t"\
    "vmov.i8 %q[cq09],#0; vmov.i8 %q[cq10],#0\n\t"\
    "vmov.i8 %q[cq11],#0; vmov.i8 %q[cq12],#0\n\t"\
    "cmp %[k_left],#2; blt 4f\n\t"\
    "vldr d0,[%[a_ptr]]; ldr r0,[%[a_ptr],#8]\n\t"\
    "ldr r1,[%[a_ptr],#12]; add %[a_ptr],%[a_ptr],#24\n\t"\
    "vldr d4,[%[b_ptr]]; vldr d5,[%[b_ptr],#8]; ldr r2,[%[b_ptr],#16]\n\t"\
    "ldr r3,[%[b_ptr],#20]; add %[b_ptr],%[b_ptr],#32\n\t"\
    "cmp %[k_left],#6; blt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    "vmov d6,r2,r3; vldr d7,[%[b_ptr],#-8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d4,d0[0]; ldr r2,[%[b_ptr]]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d5,d0[0]; ldr r3,[%[b_ptr],#4]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d4,d0[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d5,d0[1]\n\t"\
    "vmov d1,r0,r1; vldr d2,[%[a_ptr],#-8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d4,d0[2]; ldr r0,[%[a_ptr]]\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d5,d0[2]; ldr r1,[%[a_ptr],#4]\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d4,d0[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d5,d0[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d4,d1[0]; pld [%[a_ptr],#128]\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d5,d1[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d4,d1[1]; pld [%[b_ptr],#128]\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d5,d1[1]\n\t"\
    "vmov d4,r2,r3; vldr d5,[%[b_ptr],#8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d6,d1[2]; ldr r2,[%[b_ptr],#16]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d7,d1[2]; ldr r3,[%[b_ptr],#20]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d6,d1[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d7,d1[3]\n\t"\
    "vmov d0,r0,r1; vldr d1,[%[a_ptr],#8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d6,d2[0]; ldr r0,[%[a_ptr],#16]\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d7,d2[0]; ldr r1,[%[a_ptr],#20]\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d6,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d7,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d6,d2[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d7,d2[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d6,d2[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d7,d2[3]\n\t"\
    "vmov d6,r2,r3; vldr d7,[%[b_ptr],#24]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d4,d0[0]; ldr r2,[%[b_ptr],#32]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d5,d0[0]; ldr r3,[%[b_ptr],#36]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d4,d0[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d5,d0[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d4,d0[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d5,d0[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d4,d0[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d5,d0[3]\n\t"\
    "vmov d2,r0,r1; vldr d0,[%[a_ptr],#24]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d4,d1[0]; ldr r0,[%[a_ptr],#32]\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d5,d1[0]; ldr r1,[%[a_ptr],#36]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d4,d1[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d5,d1[1]\n\t"\
    "vmov d4,r2,r3; vldr d5,[%[b_ptr],#40]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d6,d1[2]; ldr r2,[%[b_ptr],#48]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d7,d1[2]; ldr r3,[%[b_ptr],#52]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d6,d1[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d7,d1[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d6,d2[0]; add %[a_ptr],%[a_ptr],#48\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d7,d2[0]; add %[b_ptr],%[b_ptr],#64\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d6,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d7,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d6,d2[2]; sub %[k_left],%[k_left],#4\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d7,d2[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d6,d2[3]; cmp %[k_left],#6\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d7,d2[3]; bge 1b\n\t"\
    "2:\n\t"\
    "cmp %[k_left],#4; blt 3f\n\t"\
    "vmov d6,r2,r3; vldr d7,[%[b_ptr],#-8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d4,d0[0]; ldr r2,[%[b_ptr]]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d5,d0[0]; ldr r3,[%[b_ptr],#4]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d4,d0[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d5,d0[1]\n\t"\
    "vmov d1,r0,r1; vldr d2,[%[a_ptr],#-8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d4,d0[2]; ldr r0,[%[a_ptr]]\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d5,d0[2]; ldr r1,[%[a_ptr],#4]\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d4,d0[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d5,d0[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d4,d1[0]; pld [%[a_ptr],#128]\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d5,d1[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d4,d1[1]; pld [%[b_ptr],#128]\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d5,d1[1]\n\t"\
    "vmov d4,r2,r3; vldr d5,[%[b_ptr],#8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d6,d1[2]; ldr r2,[%[b_ptr],#16]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d7,d1[2]; ldr r3,[%[b_ptr],#20]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d6,d1[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d7,d1[3]\n\t"\
    "vmov d0,r0,r1; vldr d1,[%[a_ptr],#8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d6,d2[0]; ldr r0,[%[a_ptr],#16]\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d7,d2[0]; ldr r1,[%[a_ptr],#20]\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d6,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d7,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d6,d2[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d7,d2[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d6,d2[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d7,d2[3]\n\t"\
    "vmov d6,r2,r3; vldr d7,[%[b_ptr],#24]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d4,d0[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d5,d0[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d4,d0[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d5,d0[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d4,d0[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d5,d0[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d4,d0[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d5,d0[3]\n\t"\
    "vmov d2,r0,r1\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d4,d1[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d5,d1[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d4,d1[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d5,d1[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d6,d1[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d7,d1[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d6,d1[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d7,d1[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d6,d2[0]; add %[a_ptr],%[a_ptr],#24\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d7,d2[0]; add %[b_ptr],%[b_ptr],#32\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d6,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d7,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d6,d2[2]; sub %[k_left],%[k_left],#4\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d7,d2[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d6,d2[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d7,d2[3]; b 4f\n\t"\
    "3:\n\t"\
    "vmov d6,r2,r3; vldr d7,[%[b_ptr],#-8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d4,d0[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d5,d0[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d4,d0[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d5,d0[1]\n\t"\
    "vmov d1,r0,r1; vldr d2,[%[a_ptr],#-8]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d4,d0[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d5,d0[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d4,d0[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d5,d0[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d4,d1[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d5,d1[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d4,d1[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d5,d1[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d6,d1[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq02],d7,d1[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d6,d1[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq04],d7,d1[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d6,d2[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq06],d7,d2[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d6,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq08],d7,d2[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d6,d2[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq10],d7,d2[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d6,d2[3]; sub %[k_left],%[k_left],#2\n\t"\
    ""ASM_VMLAL_I16" %q[cq12],d7,d2[3]\n\t"\
    "4:\n\t"\
    "cmp %[k_left],#1; blt 5f\n\t"\
    "vldr d4,[%[b_ptr]]; vldr d5,[%[b_ptr],#8]; add %[b_ptr],%[b_ptr],#16\n\t"\
    "vldr d0,[%[a_ptr]]; vldr s2,[%[a_ptr],#8]\n\t"\
    "add %[a_ptr],%[a_ptr],#12\n\t"\
    ""ASM_VMLAL_I16" %q[cq01],d4,d0[0]; "ASM_VMLAL_I16" %q[cq02],d5,d0[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq03],d4,d0[1]; "ASM_VMLAL_I16" %q[cq04],d5,d0[1]\n\t"\
    ""ASM_VMLAL_I16" %q[cq05],d4,d0[2]; "ASM_VMLAL_I16" %q[cq06],d5,d0[2]\n\t"\
    ""ASM_VMLAL_I16" %q[cq07],d4,d0[3]; "ASM_VMLAL_I16" %q[cq08],d5,d0[3]\n\t"\
    ""ASM_VMLAL_I16" %q[cq09],d4,d1[0]; "ASM_VMLAL_I16" %q[cq10],d5,d1[0]\n\t"\
    ""ASM_VMLAL_I16" %q[cq11],d4,d1[1]; "ASM_VMLAL_I16" %q[cq12],d5,d1[1]\n\t"\
    "sub %[k_left],%[k_left],#1\n\t"\
    "5:\n\t"\
   :[a_ptr]"+r"(a_ptr), [b_ptr]"+r"(b_ptr), [k_left]"+r"(k_left),\
   [cq01]"=w"(cq01), [cq02]"=w"(cq02), [cq03]"=w"(cq03), [cq04]"=w"(cq04),\
   [cq05]"=w"(cq05), [cq06]"=w"(cq06), [cq07]"=w"(cq07), [cq08]"=w"(cq08),\
   [cq09]"=w"(cq09), [cq10]"=w"(cq10), [cq11]"=w"(cq11), [cq12]"=w"(cq12)\
   ::"r0","r1","r2","r3","cc","memory","q0","q1","q2","q3");

static inline void pldw_c_6(const I32 *c) {
  __asm__("pld [%0]; pld [%0,#20]\n\t"::"r"(c):);
}

static inline void pldw_c_8(const I32 *c) {
  __asm__("pld [%0]; pld [%0,#28]\n\t"::"r"(c):);
}

#define KERNEL_M6N8 \
  I32 *c_pref = c_ptr;\
  pldw_c_6(c_pref); c_pref += ldc;\
  pldw_c_6(c_pref); c_pref += ldc;\
  pldw_c_6(c_pref); c_pref += ldc;\
  pldw_c_6(c_pref); c_pref += ldc;\
  pldw_c_6(c_pref); c_pref += ldc;\
  pldw_c_6(c_pref); c_pref += ldc;\
  pldw_c_6(c_pref); c_pref += ldc;\
  pldw_c_6(c_pref);\
  KERNEL_M6N8_UNIT(a_head, b_head)

#define KERNEL_M8N6 \
  I32 *c_pref = c_ptr;\
  pldw_c_8(c_pref); c_pref += ldc;\
  pldw_c_8(c_pref); c_pref += ldc;\
  pldw_c_8(c_pref); c_pref += ldc;\
  pldw_c_8(c_pref); c_pref += ldc;\
  pldw_c_8(c_pref); c_pref += ldc;\
  pldw_c_8(c_pref);\
  KERNEL_M6N8_UNIT(b_head, a_head)

#define SAVE_M6N8 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M6N4(cq01, cq03, cq05, cq07, cq09, cq11)\
  UNIT_SAVE_M6N4(cq02, cq04, cq06, cq08, cq10, cq12)

#define SAVE_M8N6 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M8N2(cq01, cq02, cq03, cq04)\
  UNIT_SAVE_M8N2(cq05, cq06, cq07, cq08)\
  UNIT_SAVE_M8N2(cq09, cq10, cq11, cq12)

IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(6, 8, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 6, I16, I32)

#endif
