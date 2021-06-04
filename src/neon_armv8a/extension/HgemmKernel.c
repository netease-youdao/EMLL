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


#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "common/CommonKernel.h"
#include "arm_neon/ARMCpuType.h"
#include <sched.h>
#include <arm_neon.h>

static inline void pref_c(float16_t *dat) {
  __asm__ ("prfm pstl1keep,[%0]\n\t"::"r"(dat):);
}

#define PREF_N1 pref_c(c_pref); c_pref += ldc;
#define PREF_N2 PREF_N1 PREF_N1
#define PREF_N4 PREF_N2 PREF_N2
#define PREF_N8 PREF_N4 PREF_N4
#define PREF_N16 PREF_N8 PREF_N8

#define DECLARE_C_8X8 \
  float16x8_t cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;

#define DECLARE_C_8X16 DECLARE_C_8X8 \
  float16x8_t cq09, cq10, cq11, cq12, cq13, cq14, cq15, cq16;

/* fp16-fma kernel for general out-of-order ARM processors */
/* q0 and q1 for holding data from matrix A */
/* q2 and q3 for holding data from matrix B */
#define KERNEL_M8N16_A76 \
  DECLARE_C_8X16\
  float16_t *c_pref = c_ptr + 7; PREF_N16\
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  uint32_t k_left = K;\
  __asm__ __volatile__(\
    "movi %0.16b,#0; movi %1.16b,#0\n\t"\
    "mov %2.16b,%0.16b; mov %3.16b,%1.16b\n\t"\
    "mov %4.16b,%0.16b; mov %5.16b,%1.16b\n\t"\
    "mov %6.16b,%0.16b; mov %7.16b,%1.16b\n\t"\
    "mov %8.16b,%0.16b; mov %9.16b,%1.16b\n\t"\
    "mov %10.16b,%0.16b; mov %11.16b,%1.16b\n\t"\
    "mov %12.16b,%0.16b; mov %13.16b,%1.16b\n\t"\
    "mov %14.16b,%0.16b; mov %15.16b,%1.16b\n\t"\
    "cmp %w16,#0; b.eq 004f\n\t"\
    "ldr q0,[%17],#16; ldr q2,[%18],#16; ldr q3,[%18],#16\n\t"\
    "cmp %w16,#2; b.le 002f\n\t"\
    "001:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "ldr q1,[%17],#32\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v0.8h,v2.h[4]; fmla %5.8h,v0.8h,v2.h[5]\n\t"\
    "prfm pldl1keep,[%17,#128]\n\t"\
    "fmla %6.8h,v0.8h,v2.h[6]; fmla %7.8h,v0.8h,v2.h[7]\n\t"\
    "ldr q2,[%18],#64\n\t"\
    "fmla %8.8h,v0.8h,v3.h[0]; fmla %9.8h,v0.8h,v3.h[1]\n\t"\
    "fmla %10.8h,v0.8h,v3.h[2]; fmla %11.8h,v0.8h,v3.h[3]\n\t"\
    "sub %w16,%w16,#2\n\t"\
    "fmla %12.8h,v0.8h,v3.h[4]; fmla %13.8h,v0.8h,v3.h[5]\n\t"\
    "fmla %14.8h,v0.8h,v3.h[6]; fmla %15.8h,v0.8h,v3.h[7]\n\t"\
    "ldr q3,[%18,#-48]\n\t"\
    "fmla %0.8h,v1.8h,v2.h[0]; fmla %1.8h,v1.8h,v2.h[1]\n\t"\
    "ldr q0,[%17,#-16]\n\t"\
    "fmla %2.8h,v1.8h,v2.h[2]; fmla %3.8h,v1.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v1.8h,v2.h[4]; fmla %5.8h,v1.8h,v2.h[5]\n\t"\
    "cmp %w16,#2\n\t"\
    "fmla %6.8h,v1.8h,v2.h[6]; fmla %7.8h,v1.8h,v2.h[7]\n\t"\
    "ldr q2,[%18,#-32]\n\t"\
    "fmla %8.8h,v1.8h,v3.h[0]; fmla %9.8h,v1.8h,v3.h[1]\n\t"\
    "fmla %10.8h,v1.8h,v3.h[2]; fmla %11.8h,v1.8h,v3.h[3]\n\t"\
    "fmla %12.8h,v1.8h,v3.h[4]; fmla %13.8h,v1.8h,v3.h[5]\n\t"\
    "fmla %14.8h,v1.8h,v3.h[6]; fmla %15.8h,v1.8h,v3.h[7]\n\t"\
    "ldr q3,[%18,#-16]; b.gt 001b\n\t"\
    "002:\n\t"\
    "cmp %w16,#2; b.ne 003f\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "ldr q1,[%17],#16\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v0.8h,v2.h[4]; fmla %5.8h,v0.8h,v2.h[5]\n\t"\
    "fmla %6.8h,v0.8h,v2.h[6]; fmla %7.8h,v0.8h,v2.h[7]\n\t"\
    "ldr q2,[%18],#32\n\t"\
    "fmla %8.8h,v0.8h,v3.h[0]; fmla %9.8h,v0.8h,v3.h[1]\n\t"\
    "fmla %10.8h,v0.8h,v3.h[2]; fmla %11.8h,v0.8h,v3.h[3]\n\t"\
    "sub %w16,%w16,#2\n\t"\
    "fmla %12.8h,v0.8h,v3.h[4]; fmla %13.8h,v0.8h,v3.h[5]\n\t"\
    "fmla %14.8h,v0.8h,v3.h[6]; fmla %15.8h,v0.8h,v3.h[7]\n\t"\
    "ldr q3,[%18,#-16]\n\t"\
    "fmla %0.8h,v1.8h,v2.h[0]; fmla %1.8h,v1.8h,v2.h[1]\n\t"\
    "fmla %2.8h,v1.8h,v2.h[2]; fmla %3.8h,v1.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v1.8h,v2.h[4]; fmla %5.8h,v1.8h,v2.h[5]\n\t"\
    "fmla %6.8h,v1.8h,v2.h[6]; fmla %7.8h,v1.8h,v2.h[7]\n\t"\
    "fmla %8.8h,v1.8h,v3.h[0]; fmla %9.8h,v1.8h,v3.h[1]\n\t"\
    "fmla %10.8h,v1.8h,v3.h[2]; fmla %11.8h,v1.8h,v3.h[3]\n\t"\
    "fmla %12.8h,v1.8h,v3.h[4]; fmla %13.8h,v1.8h,v3.h[5]\n\t"\
    "fmla %14.8h,v1.8h,v3.h[6]; fmla %15.8h,v1.8h,v3.h[7]\n\t"\
    "b 004f\n\t"\
    "003:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v0.8h,v2.h[4]; fmla %5.8h,v0.8h,v2.h[5]\n\t"\
    "fmla %6.8h,v0.8h,v2.h[6]; fmla %7.8h,v0.8h,v2.h[7]\n\t"\
    "fmla %8.8h,v0.8h,v3.h[0]; fmla %9.8h,v0.8h,v3.h[1]\n\t"\
    "fmla %10.8h,v0.8h,v3.h[2]; fmla %11.8h,v0.8h,v3.h[3]\n\t"\
    "sub %w16,%w16,#1\n\t"\
    "fmla %12.8h,v0.8h,v3.h[4]; fmla %13.8h,v0.8h,v3.h[5]\n\t"\
    "fmla %14.8h,v0.8h,v3.h[6]; fmla %15.8h,v0.8h,v3.h[7]\n\t"\
    "004:\n\t"\
  :"=w"(cq01),"=w"(cq02),"=w"(cq03),"=w"(cq04)\
  ,"=w"(cq05),"=w"(cq06),"=w"(cq07),"=w"(cq08)\
  ,"=w"(cq09),"=w"(cq10),"=w"(cq11),"=w"(cq12)\
  ,"=w"(cq13),"=w"(cq14),"=w"(cq15),"=w"(cq16)\
  ,"+r"(k_left),"+r"(a_ptr),"+r"(b_ptr1)\
  ::"cc","memory","v0","v1","v2","v3");

#define KERNEL_M16N8_A76 \
  DECLARE_C_8X16\
  float16_t *c_pref = c_ptr + 15; PREF_N8\
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  uint32_t k_left = K;\
  __asm__ __volatile__(\
    "movi %0.16b,#0; movi %1.16b,#0\n\t"\
    "mov %2.16b,%0.16b; mov %3.16b,%1.16b\n\t"\
    "mov %4.16b,%0.16b; mov %5.16b,%1.16b\n\t"\
    "mov %6.16b,%0.16b; mov %7.16b,%1.16b\n\t"\
    "mov %8.16b,%0.16b; mov %9.16b,%1.16b\n\t"\
    "mov %10.16b,%0.16b; mov %11.16b,%1.16b\n\t"\
    "mov %12.16b,%0.16b; mov %13.16b,%1.16b\n\t"\
    "mov %14.16b,%0.16b; mov %15.16b,%1.16b\n\t"\
    "cmp %w16,#0; b.eq 004f\n\t"\
    "ldr q0,[%17],#32; ldr q2,[%18],#16; ldr q1,[%17,#-16]\n\t"\
    "cmp %w16,#2; b.le 002f\n\t"\
    "001:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %2.8h,v0.8h,v2.h[1]\n\t"\
    "ldr q3,[%18],#32\n\t"\
    "fmla %4.8h,v0.8h,v2.h[2]; fmla %6.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v2.h[4]; fmla %10.8h,v0.8h,v2.h[5]\n\t"\
    "prfm pldl1keep,[%18,#128]\n\t"\
    "fmla %12.8h,v0.8h,v2.h[6]; fmla %14.8h,v0.8h,v2.h[7]\n\t"\
    "ldr q0,[%17],#64\n\t"\
    "fmla %1.8h,v1.8h,v2.h[0]; fmla %3.8h,v1.8h,v2.h[1]\n\t"\
    "fmla %5.8h,v1.8h,v2.h[2]; fmla %7.8h,v1.8h,v2.h[3]\n\t"\
    "sub %w16,%w16,#2\n\t"\
    "fmla %9.8h,v1.8h,v2.h[4]; fmla %11.8h,v1.8h,v2.h[5]\n\t"\
    "fmla %13.8h,v1.8h,v2.h[6]; fmla %15.8h,v1.8h,v2.h[7]\n\t"\
    "ldr q1,[%17,#-48]\n\t"\
    "fmla %0.8h,v0.8h,v3.h[0]; fmla %2.8h,v0.8h,v3.h[1]\n\t"\
    "ldr q2,[%18,#-16]\n\t"\
    "fmla %4.8h,v0.8h,v3.h[2]; fmla %6.8h,v0.8h,v3.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v3.h[4]; fmla %10.8h,v0.8h,v3.h[5]\n\t"\
    "cmp %w16,#2\n\t"\
    "fmla %12.8h,v0.8h,v3.h[6]; fmla %14.8h,v0.8h,v3.h[7]\n\t"\
    "ldr q0,[%17,#-32]\n\t"\
    "fmla %1.8h,v1.8h,v3.h[0]; fmla %3.8h,v1.8h,v3.h[1]\n\t"\
    "fmla %5.8h,v1.8h,v3.h[2]; fmla %7.8h,v1.8h,v3.h[3]\n\t"\
    "fmla %9.8h,v1.8h,v3.h[4]; fmla %11.8h,v1.8h,v3.h[5]\n\t"\
    "fmla %13.8h,v1.8h,v3.h[6]; fmla %15.8h,v1.8h,v3.h[7]\n\t"\
    "ldr q1,[%17,#-16]; b.gt 001b\n\t"\
    "002:\n\t"\
    "cmp %w16,#2; b.ne 003f\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %2.8h,v0.8h,v2.h[1]\n\t"\
    "ldr q3,[%18],#16\n\t"\
    "fmla %4.8h,v0.8h,v2.h[2]; fmla %6.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v2.h[4]; fmla %10.8h,v0.8h,v2.h[5]\n\t"\
    "fmla %12.8h,v0.8h,v2.h[6]; fmla %14.8h,v0.8h,v2.h[7]\n\t"\
    "ldr q0,[%17],#32\n\t"\
    "fmla %1.8h,v1.8h,v2.h[0]; fmla %3.8h,v1.8h,v2.h[1]\n\t"\
    "fmla %5.8h,v1.8h,v2.h[2]; fmla %7.8h,v1.8h,v2.h[3]\n\t"\
    "sub %w16,%w16,#2\n\t"\
    "fmla %9.8h,v1.8h,v2.h[4]; fmla %11.8h,v1.8h,v2.h[5]\n\t"\
    "fmla %13.8h,v1.8h,v2.h[6]; fmla %15.8h,v1.8h,v2.h[7]\n\t"\
    "ldr q1,[%17,#-16]\n\t"\
    "fmla %0.8h,v0.8h,v3.h[0]; fmla %2.8h,v0.8h,v3.h[1]\n\t"\
    "fmla %4.8h,v0.8h,v3.h[2]; fmla %6.8h,v0.8h,v3.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v3.h[4]; fmla %10.8h,v0.8h,v3.h[5]\n\t"\
    "fmla %12.8h,v0.8h,v3.h[6]; fmla %14.8h,v0.8h,v3.h[7]\n\t"\
    "fmla %1.8h,v1.8h,v3.h[0]; fmla %3.8h,v1.8h,v3.h[1]\n\t"\
    "fmla %5.8h,v1.8h,v3.h[2]; fmla %7.8h,v1.8h,v3.h[3]\n\t"\
    "fmla %9.8h,v1.8h,v3.h[4]; fmla %11.8h,v1.8h,v3.h[5]\n\t"\
    "fmla %13.8h,v1.8h,v3.h[6]; fmla %15.8h,v1.8h,v3.h[7]\n\t"\
    "b 004f\n\t"\
    "003:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %2.8h,v0.8h,v2.h[1]\n\t"\
    "fmla %4.8h,v0.8h,v2.h[2]; fmla %6.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v2.h[4]; fmla %10.8h,v0.8h,v2.h[5]\n\t"\
    "fmla %12.8h,v0.8h,v2.h[6]; fmla %14.8h,v0.8h,v2.h[7]\n\t"\
    "fmla %1.8h,v1.8h,v2.h[0]; fmla %3.8h,v1.8h,v2.h[1]\n\t"\
    "fmla %5.8h,v1.8h,v2.h[2]; fmla %7.8h,v1.8h,v2.h[3]\n\t"\
    "sub %w16,%w16,#1\n\t"\
    "fmla %9.8h,v1.8h,v2.h[4]; fmla %11.8h,v1.8h,v2.h[5]\n\t"\
    "fmla %13.8h,v1.8h,v2.h[6]; fmla %15.8h,v1.8h,v2.h[7]\n\t"\
    "004:\n\t"\
  :"=w"(cq01),"=w"(cq02),"=w"(cq03),"=w"(cq04)\
  ,"=w"(cq05),"=w"(cq06),"=w"(cq07),"=w"(cq08)\
  ,"=w"(cq09),"=w"(cq10),"=w"(cq11),"=w"(cq12)\
  ,"=w"(cq13),"=w"(cq14),"=w"(cq15),"=w"(cq16)\
  ,"+r"(k_left),"+r"(a_ptr),"+r"(b_ptr1)\
  ::"cc","memory","v0","v1","v2","v3");

#define KERNEL_M8N8_A76 \
  DECLARE_C_8X8\
  float16_t *c_pref = c_ptr + 7; PREF_N8\
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  uint32_t k_left = K;\
  __asm__ __volatile__(\
    "movi %0.16b,#0; movi %1.16b,#0\n\t"\
    "mov %2.16b,%0.16b; mov %3.16b,%1.16b\n\t"\
    "mov %4.16b,%0.16b; mov %5.16b,%1.16b\n\t"\
    "mov %6.16b,%0.16b; mov %7.16b,%1.16b\n\t"\
    "cmp %w8,#0; b.eq 104f\n\t"\
    "ldr q0,[%9],#16; ldr q2,[%10],#16\n\t"\
    "cmp %w8,#2; b.le 102f\n\t"\
    "101:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "ldr q1,[%9],#32\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "ldr q3,[%10],#32\n\t"\
    "fmla %4.8h,v0.8h,v2.h[4]; fmla %5.8h,v0.8h,v2.h[5]\n\t"\
    "prfm pldl1keep,[%9,#128]\n\t"\
    "fmla %6.8h,v0.8h,v2.h[6]; fmla %7.8h,v0.8h,v2.h[7]\n\t"\
    "ldr q0,[%9,#-16]\n\t"\
    "fmla %0.8h,v1.8h,v3.h[0]; fmla %1.8h,v1.8h,v3.h[1]\n\t"\
    "ldr q2,[%10,#-16]\n\t"\
    "fmla %2.8h,v1.8h,v3.h[2]; fmla %3.8h,v1.8h,v3.h[3]\n\t"\
    "sub %w8,%w8,#2\n\t"\
    "fmla %4.8h,v1.8h,v3.h[4]; fmla %5.8h,v1.8h,v3.h[5]\n\t"\
    "cmp %w8,#2\n\t"\
    "fmla %6.8h,v1.8h,v3.h[6]; fmla %7.8h,v1.8h,v3.h[7]\n\t"\
    "b.gt 101b\n\t"\
    "102:\n\t"\
    "cmp %w8,#2; b.ne 103f\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "ldr q1,[%9],#16\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "ldr q3,[%10],#16\n\t"\
    "fmla %4.8h,v0.8h,v2.h[4]; fmla %5.8h,v0.8h,v2.h[5]\n\t"\
    "prfm pldl1keep,[%9,#128]\n\t"\
    "fmla %6.8h,v0.8h,v2.h[6]; fmla %7.8h,v0.8h,v2.h[7]\n\t"\
    "fmla %0.8h,v1.8h,v3.h[0]; fmla %1.8h,v1.8h,v3.h[1]\n\t"\
    "fmla %2.8h,v1.8h,v3.h[2]; fmla %3.8h,v1.8h,v3.h[3]\n\t"\
    "sub %w8,%w8,#2\n\t"\
    "fmla %4.8h,v1.8h,v3.h[4]; fmla %5.8h,v1.8h,v3.h[5]\n\t"\
    "fmla %6.8h,v1.8h,v3.h[6]; fmla %7.8h,v1.8h,v3.h[7]\n\t"\
    "b 104f\n\t"\
    "103:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v0.8h,v2.h[4]; fmla %5.8h,v0.8h,v2.h[5]\n\t"\
    "fmla %6.8h,v0.8h,v2.h[6]; fmla %7.8h,v0.8h,v2.h[7]\n\t"\
    "sub %w8,%w8,#1\n\t"\
    "104:\n\t"\
  :"=w"(cq01),"=w"(cq02),"=w"(cq03),"=w"(cq04)\
  ,"=w"(cq05),"=w"(cq06),"=w"(cq07),"=w"(cq08)\
  ,"+r"(k_left),"+r"(a_ptr),"+r"(b_ptr1)\
  ::"cc","memory","v0","v1","v2","v3");

/* fp16-fma kernel for A55 specially */
#define KERNEL_M8N16_A55 \
  DECLARE_C_8X16\
  float16_t *c_pref = c_ptr + 7; PREF_N16\
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  uint32_t k_left = K;\
  __asm__ __volatile__(\
    "movi %0.16b,#0; movi %1.16b,#0\n\t"\
    "mov %2.16b,%0.16b; mov %3.16b,%1.16b\n\t"\
    "mov %4.16b,%0.16b; mov %5.16b,%1.16b\n\t"\
    "mov %6.16b,%0.16b; mov %7.16b,%1.16b\n\t"\
    "mov %8.16b,%0.16b; mov %9.16b,%1.16b\n\t"\
    "mov %10.16b,%0.16b; mov %11.16b,%1.16b\n\t"\
    "mov %12.16b,%0.16b; mov %13.16b,%1.16b\n\t"\
    "mov %14.16b,%0.16b; mov %15.16b,%1.16b\n\t"\
    "cmp %w16,#0; b.eq 004f\n\t"\
    "ldr q0,[%17],#16; ldr d2,[%18],#32; ldr d3,[%18,#-24]\n\t"\
    "ldr d4,[%18,#-16]; ldr d5,[%18,#-8]\n\t"\
    "cmp %w16,#2; b.le 002f\n\t"\
    "001:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "ldr d1,[%17],#32\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "ldr x0,[%17,#-24]\n\t"\
    "fmla %4.8h,v0.8h,v3.h[0]\n\t"\
    "ldr d2,[%18],#64\n\t"\
    "fmla %5.8h,v0.8h,v3.h[1]\n\t"\
    "prfm pldl1keep,[%17,#128]\n\t"\
    "fmla %6.8h,v0.8h,v3.h[2]; fmla %7.8h,v0.8h,v3.h[3]\n\t"\
    "ldr d3,[%18,#-56]\n\t"\
    "fmla %8.8h,v0.8h,v4.h[0]; fmla %9.8h,v0.8h,v4.h[1]\n\t"\
    "fmov v1.d[1],x0\n\t"\
    "fmla %10.8h,v0.8h,v4.h[2]; fmla %11.8h,v0.8h,v4.h[3]\n\t"\
    "sub %w16,%w16,#2\n\t"\
    "fmla %12.8h,v0.8h,v5.h[0]\n\t"\
    "ldr d4,[%18,#-48]\n\t"\
    "fmla %13.8h,v0.8h,v5.h[1]\n\t"\
    "fmla %14.8h,v0.8h,v5.h[2]; fmla %15.8h,v0.8h,v5.h[3]\n\t"\
    "ldr d5,[%18,#-40]\n\t"\
    "fmla %0.8h,v1.8h,v2.h[0]; fmla %1.8h,v1.8h,v2.h[1]\n\t"\
    "ldr d0,[%17,#-16]\n\t"\
    "fmla %2.8h,v1.8h,v2.h[2]; fmla %3.8h,v1.8h,v2.h[3]\n\t"\
    "ldr x0,[%17,#-8]\n\t"\
    "fmla %4.8h,v1.8h,v3.h[0]\n\t"\
    "ldr d2,[%18,#-32]\n\t"\
    "fmla %5.8h,v1.8h,v3.h[1]\n\t"\
    "cmp %w16,#2\n\t"\
    "fmla %6.8h,v1.8h,v3.h[2]; fmla %7.8h,v1.8h,v3.h[3]\n\t"\
    "ldr d3,[%18,#-24]\n\t"\
    "fmla %8.8h,v1.8h,v4.h[0]; fmla %9.8h,v1.8h,v4.h[1]\n\t"\
    "fmla %10.8h,v1.8h,v4.h[2]; fmla %11.8h,v1.8h,v4.h[3]\n\t"\
    "fmov v0.d[1],x0\n\t"\
    "fmla %12.8h,v1.8h,v5.h[0]\n\t"\
    "ldr d4,[%18,#-16]\n\t"\
    "fmla %13.8h,v1.8h,v5.h[1]\n\t"\
    "fmla %14.8h,v1.8h,v5.h[2]; fmla %15.8h,v1.8h,v5.h[3]\n\t"\
    "ldr d5,[%18,#-8]; b.gt 001b\n\t"\
    "002:\n\t"\
    "cmp %w16,#2; b.ne 003f\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "ldr d1,[%17],#16\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "ldr x0,[%17,#-8]\n\t"\
    "fmla %4.8h,v0.8h,v3.h[0]\n\t"\
    "ldr d2,[%18],#32\n\t"\
    "fmla %5.8h,v0.8h,v3.h[1]\n\t"\
    "fmla %6.8h,v0.8h,v3.h[2]; fmla %7.8h,v0.8h,v3.h[3]\n\t"\
    "ldr d3,[%18,#-24]\n\t"\
    "fmla %8.8h,v0.8h,v4.h[0]; fmla %9.8h,v0.8h,v4.h[1]\n\t"\
    "fmla %10.8h,v0.8h,v4.h[2]; fmla %11.8h,v0.8h,v4.h[3]\n\t"\
    "sub %w16,%w16,#2\n\t"\
    "fmla %12.8h,v0.8h,v5.h[0]\n\t"\
    "ldr d4,[%18,#-16]\n\t"\
    "fmla %13.8h,v0.8h,v5.h[1]\n\t"\
    "fmov v1.d[1],x0\n\t"\
    "fmla %14.8h,v0.8h,v5.h[2]; fmla %15.8h,v0.8h,v5.h[3]\n\t"\
    "ldr d5,[%18,#-8]\n\t"\
    "fmla %0.8h,v1.8h,v2.h[0]; fmla %1.8h,v1.8h,v2.h[1]\n\t"\
    "fmla %2.8h,v1.8h,v2.h[2]; fmla %3.8h,v1.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v1.8h,v3.h[0]; fmla %5.8h,v1.8h,v3.h[1]\n\t"\
    "fmla %6.8h,v1.8h,v3.h[2]; fmla %7.8h,v1.8h,v3.h[3]\n\t"\
    "fmla %8.8h,v1.8h,v4.h[0]; fmla %9.8h,v1.8h,v4.h[1]\n\t"\
    "fmla %10.8h,v1.8h,v4.h[2]; fmla %11.8h,v1.8h,v4.h[3]\n\t"\
    "fmla %12.8h,v1.8h,v5.h[0]; fmla %13.8h,v1.8h,v5.h[1]\n\t"\
    "fmla %14.8h,v1.8h,v5.h[2]; fmla %15.8h,v1.8h,v5.h[3]\n\t"\
    "b 004f\n\t"\
    "003:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v0.8h,v3.h[0]; fmla %5.8h,v0.8h,v3.h[1]\n\t"\
    "fmla %6.8h,v0.8h,v3.h[2]; fmla %7.8h,v0.8h,v3.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v4.h[0]; fmla %9.8h,v0.8h,v4.h[1]\n\t"\
    "fmla %10.8h,v0.8h,v4.h[2]; fmla %11.8h,v0.8h,v4.h[3]\n\t"\
    "sub %w16,%w16,#1\n\t"\
    "fmla %12.8h,v0.8h,v5.h[0]; fmla %13.8h,v0.8h,v5.h[1]\n\t"\
    "fmla %14.8h,v0.8h,v5.h[2]; fmla %15.8h,v0.8h,v5.h[3]\n\t"\
    "004:\n\t"\
  :"=w"(cq01),"=w"(cq02),"=w"(cq03),"=w"(cq04)\
  ,"=w"(cq05),"=w"(cq06),"=w"(cq07),"=w"(cq08)\
  ,"=w"(cq09),"=w"(cq10),"=w"(cq11),"=w"(cq12)\
  ,"=w"(cq13),"=w"(cq14),"=w"(cq15),"=w"(cq16)\
  ,"+r"(k_left),"+r"(a_ptr),"+r"(b_ptr1)\
  ::"cc","memory","v0","v1","v2","v3","v4","v5","x0");

#define KERNEL_M16N8_A55 \
  DECLARE_C_8X16\
  float16_t *c_pref = c_ptr + 15; PREF_N8\
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  uint32_t k_left = K;\
  __asm__ __volatile__(\
    "movi %0.16b,#0; movi %1.16b,#0\n\t"\
    "mov %2.16b,%0.16b; mov %3.16b,%1.16b\n\t"\
    "mov %4.16b,%0.16b; mov %5.16b,%1.16b\n\t"\
    "mov %6.16b,%0.16b; mov %7.16b,%1.16b\n\t"\
    "mov %8.16b,%0.16b; mov %9.16b,%1.16b\n\t"\
    "mov %10.16b,%0.16b; mov %11.16b,%1.16b\n\t"\
    "mov %12.16b,%0.16b; mov %13.16b,%1.16b\n\t"\
    "mov %14.16b,%0.16b; mov %15.16b,%1.16b\n\t"\
    "cmp %w16,#0; b.eq 004f\n\t"\
    "ldr q0,[%17],#32\n\t"\
    "ldr d2,[%18],#16; ldr d3,[%18,#-8]\n\t"\
    "cmp %w16,#2; b.le 002f\n\t"\
    "001:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]\n\t"\
    "fmla %2.8h,v0.8h,v2.h[1]; ldr d1,[%17,#-16]\n\t"\
    "fmla %4.8h,v0.8h,v2.h[2]; ldr x0,[%17,#-8]\n\t"\
    "fmla %6.8h,v0.8h,v2.h[3]; prfm pldl2keep,[%18,#128]\n\t"\
    "fmla %8.8h,v0.8h,v3.h[0]; ldr d4,[%18],#32\n\t"\
    "fmla %10.8h,v0.8h,v3.h[1]; fmov v1.d[1],x0\n\t"\
    "fmla %12.8h,v0.8h,v3.h[2]\n\t"\
    "fmla %14.8h,v0.8h,v3.h[3]; ldr d5,[%18,#-24]\n\t"\
    "fmla %1.8h,v1.8h,v2.h[0]; ldr d0,[%17],#64\n\t"\
    "fmla %3.8h,v1.8h,v2.h[1]\n\t"\
    "fmla %5.8h,v1.8h,v2.h[2]; ldr x0,[%17,#-56]\n\t"\
    "fmla %7.8h,v1.8h,v2.h[3]\n\t"\
    "fmla %9.8h,v1.8h,v3.h[0]\n\t"\
    "fmla %11.8h,v1.8h,v3.h[1]; fmov v0.d[1],x0\n\t"\
    "fmla %13.8h,v1.8h,v3.h[2]\n\t"\
    "fmla %15.8h,v1.8h,v3.h[3]\n\t"\
    "fmla %0.8h,v0.8h,v4.h[0]; ldr d1,[%17,#-48]\n\t"\
    "fmla %2.8h,v0.8h,v4.h[1]; ldr x0,[%17,#-40]\n\t"\
    "fmla %4.8h,v0.8h,v4.h[2]; ldr d2,[%18,#-16]\n\t"\
    "fmla %6.8h,v0.8h,v4.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v5.h[0]\n\t"\
    "fmla %10.8h,v0.8h,v5.h[1]; fmov v1.d[1],x0\n\t"\
    "fmla %12.8h,v0.8h,v5.h[2]; ldr d3,[%18,#-8]\n\t"\
    "fmla %14.8h,v0.8h,v5.h[3]\n\t"\
    "fmla %1.8h,v1.8h,v4.h[0]; ldr d0,[%17,#-32]\n\t"\
    "fmla %3.8h,v1.8h,v4.h[1]; ldr x0,[%17,#-24]\n\t"\
    "fmla %5.8h,v1.8h,v4.h[2]\n\t"\
    "fmla %7.8h,v1.8h,v4.h[3]; sub %w16,%w16,#2\n\t"\
    "fmla %9.8h,v1.8h,v5.h[0]\n\t"\
    "fmla %11.8h,v1.8h,v5.h[1]; fmov v0.d[1],x0\n\t"\
    "fmla %13.8h,v1.8h,v5.h[2]; cmp %w16,#2\n\t"\
    "fmla %15.8h,v1.8h,v5.h[3]; b.gt 001b\n\t"\
    "002:\n\t"\
    "cmp %w16,#2; b.ne 003f\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]\n\t"\
    "fmla %2.8h,v0.8h,v2.h[1]; ldr d1,[%17,#-16]\n\t"\
    "fmla %4.8h,v0.8h,v2.h[2]; ldr x0,[%17,#-8]\n\t"\
    "fmla %6.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v3.h[0]; ldr d4,[%18],#16\n\t"\
    "fmla %10.8h,v0.8h,v3.h[1]; fmov v1.d[1],x0\n\t"\
    "fmla %12.8h,v0.8h,v3.h[2]\n\t"\
    "fmla %14.8h,v0.8h,v3.h[3]; ldr d5,[%18,#-8]\n\t"\
    "fmla %1.8h,v1.8h,v2.h[0]; ldr d0,[%17],#32\n\t"\
    "fmla %3.8h,v1.8h,v2.h[1]\n\t"\
    "fmla %5.8h,v1.8h,v2.h[2]; ldr x0,[%17,#-24]\n\t"\
    "fmla %7.8h,v1.8h,v2.h[3]\n\t"\
    "fmla %9.8h,v1.8h,v3.h[0]\n\t"\
    "fmla %11.8h,v1.8h,v3.h[1]; fmov v0.d[1],x0\n\t"\
    "fmla %13.8h,v1.8h,v3.h[2]\n\t"\
    "fmla %15.8h,v1.8h,v3.h[3]\n\t"\
    "fmla %0.8h,v0.8h,v4.h[0]; ldr d1,[%17,#-16]\n\t"\
    "fmla %2.8h,v0.8h,v4.h[1]; ldr x0,[%17,#-8]\n\t"\
    "fmla %4.8h,v0.8h,v4.h[2]\n\t"\
    "fmla %6.8h,v0.8h,v4.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v5.h[0]\n\t"\
    "fmla %10.8h,v0.8h,v5.h[1]; fmov v1.d[1],x0\n\t"\
    "fmla %12.8h,v0.8h,v5.h[2]\n\t"\
    "fmla %14.8h,v0.8h,v5.h[3]\n\t"\
    "fmla %1.8h,v1.8h,v4.h[0]\n\t"\
    "fmla %3.8h,v1.8h,v4.h[1]\n\t"\
    "fmla %5.8h,v1.8h,v4.h[2]\n\t"\
    "fmla %7.8h,v1.8h,v4.h[3]; sub %w16,%w16,#2\n\t"\
    "fmla %9.8h,v1.8h,v5.h[0]\n\t"\
    "fmla %11.8h,v1.8h,v5.h[1]\n\t"\
    "fmla %13.8h,v1.8h,v5.h[2]\n\t"\
    "fmla %15.8h,v1.8h,v5.h[3]; b 004f\n\t"\
    "003:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]\n\t"\
    "fmla %2.8h,v0.8h,v2.h[1]; ldr d1,[%17,#-16]\n\t"\
    "fmla %4.8h,v0.8h,v2.h[2]; ldr x0,[%17,#-8]\n\t"\
    "fmla %6.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %8.8h,v0.8h,v3.h[0]\n\t"\
    "fmla %10.8h,v0.8h,v3.h[1]; fmov v1.d[1],x0\n\t"\
    "fmla %12.8h,v0.8h,v3.h[2]\n\t"\
    "fmla %14.8h,v0.8h,v3.h[3]\n\t"\
    "fmla %1.8h,v1.8h,v2.h[0]\n\t"\
    "fmla %3.8h,v1.8h,v2.h[1]\n\t"\
    "fmla %5.8h,v1.8h,v2.h[2]\n\t"\
    "fmla %7.8h,v1.8h,v2.h[3]\n\t"\
    "fmla %9.8h,v1.8h,v3.h[0]\n\t"\
    "fmla %11.8h,v1.8h,v3.h[1]; sub %w16,%w16,#1\n\t"\
    "fmla %13.8h,v1.8h,v3.h[2]\n\t"\
    "fmla %15.8h,v1.8h,v3.h[3]\n\t"\
    "004:\n\t"\
  :"=w"(cq01),"=w"(cq02),"=w"(cq03),"=w"(cq04)\
  ,"=w"(cq05),"=w"(cq06),"=w"(cq07),"=w"(cq08)\
  ,"=w"(cq09),"=w"(cq10),"=w"(cq11),"=w"(cq12)\
  ,"=w"(cq13),"=w"(cq14),"=w"(cq15),"=w"(cq16)\
  ,"+r"(k_left),"+r"(a_ptr),"+r"(b_ptr1)\
  ::"cc","memory","v0","v1","v2","v3","v4","v5","x0");

#define KERNEL_M8N8_A55 \
  DECLARE_C_8X8\
  float16_t *c_pref = c_ptr + 7; PREF_N8\
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  uint32_t k_left = K;\
  __asm__ __volatile__(\
    "movi %0.16b,#0; movi %1.16b,#0\n\t"\
    "mov %2.16b,%0.16b; mov %3.16b,%1.16b\n\t"\
    "mov %4.16b,%0.16b; mov %5.16b,%1.16b\n\t"\
    "mov %6.16b,%0.16b; mov %7.16b,%1.16b\n\t"\
    "cmp %w8,#0; b.eq 104f\n\t"\
    "ldr q0,[%9],#16; ldr d2,[%10],#16; ldr d3,[%10,#-8]\n\t"\
    "cmp %w8,#2; b.le 102f\n\t"\
    "101:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; ldr d1,[%9],#32\n\t"\
    "fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; ldr x0,[%9,#-24]\n\t"\
    "fmla %3.8h,v0.8h,v2.h[3]; prfm pldl1keep,[%9,#128]\n\t"\
    "fmla %4.8h,v0.8h,v3.h[0]; ldr d2,[%10],#32\n\t"\
    "fmla %5.8h,v0.8h,v3.h[1]; fmov v1.d[1],x0\n\t"\
    "fmla %6.8h,v0.8h,v3.h[2]\n\t"\
    "fmla %7.8h,v0.8h,v3.h[3]; ldr d3,[%10,#-24]\n\t"\
    "fmla %0.8h,v1.8h,v2.h[0]; ldr d0,[%9,#-16]\n\t"\
    "fmla %1.8h,v1.8h,v2.h[1]; ldr x0,[%9,#-8]\n\t"\
    "fmla %2.8h,v1.8h,v2.h[2]\n\t"\
    "fmla %3.8h,v1.8h,v2.h[3]; ldr d2,[%10,#-16]\n\t"\
    "fmla %4.8h,v1.8h,v3.h[0]; fmov v0.d[1],x0\n\t"\
    "fmla %5.8h,v1.8h,v3.h[1]; sub %w8,%w8,#2\n\t"\
    "fmla %6.8h,v1.8h,v3.h[2]; cmp %w8,#2\n\t"\
    "fmla %7.8h,v1.8h,v3.h[3]; ldr d3,[%10,#-8]\n\t"\
    "b.gt 101b\n\t"\
    "102:\n\t"\
    "cmp %w8,#2; b.ne 103f\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; ldr d1,[%9],#16\n\t"\
    "fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; ldr x0,[%9,#-8]\n\t"\
    "fmla %3.8h,v0.8h,v2.h[3]; prfm pldl1keep,[%9,#128]\n\t"\
    "fmla %4.8h,v0.8h,v3.h[0]; ldr d2,[%10],#16\n\t"\
    "fmla %5.8h,v0.8h,v3.h[1]; fmov v1.d[1],x0\n\t"\
    "fmla %6.8h,v0.8h,v3.h[2]\n\t"\
    "fmla %7.8h,v0.8h,v3.h[3]; ldr d3,[%10,#-8]\n\t"\
    "fmla %0.8h,v1.8h,v2.h[0]\n\t"\
    "fmla %1.8h,v1.8h,v2.h[1]\n\t"\
    "fmla %2.8h,v1.8h,v2.h[2]\n\t"\
    "fmla %3.8h,v1.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v1.8h,v3.h[0]\n\t"\
    "fmla %5.8h,v1.8h,v3.h[1]; sub %w8,%w8,#2\n\t"\
    "fmla %6.8h,v1.8h,v3.h[2]\n\t"\
    "fmla %7.8h,v1.8h,v3.h[3]\n\t"\
    "b 104f\n\t"\
    "103:\n\t"\
    "fmla %0.8h,v0.8h,v2.h[0]; fmla %1.8h,v0.8h,v2.h[1]\n\t"\
    "fmla %2.8h,v0.8h,v2.h[2]; fmla %3.8h,v0.8h,v2.h[3]\n\t"\
    "fmla %4.8h,v0.8h,v3.h[0]; fmla %5.8h,v0.8h,v3.h[1]\n\t"\
    "fmla %6.8h,v0.8h,v3.h[2]; fmla %7.8h,v0.8h,v3.h[3]\n\t"\
    "sub %w8,%w8,#1\n\t"\
    "104:\n\t"\
  :"=w"(cq01),"=w"(cq02),"=w"(cq03),"=w"(cq04)\
  ,"=w"(cq05),"=w"(cq06),"=w"(cq07),"=w"(cq08)\
  ,"+r"(k_left),"+r"(a_ptr),"+r"(b_ptr1)\
  ::"cc","memory","v0","v1","v2","v3","x0");

#define KERNEL_M8N4_UNIT(a_head, b_head) \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x8_t cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f16(0.0f);\
  cq05 = cq06 = cq07 = cq08 = vdupq_n_f16(0.0f);\
  float16x8_t aq01, aq02, bq01;\
  uint32_t k_left = K;\
  if (k_left > 1) {\
    aq01 = vld1q_f16(a_ptr);\
    aq02 = vld1q_f16(a_ptr + 8); a_ptr += 16;\
    bq01 = vld1q_f16(b_ptr1); b_ptr1 += 8;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cq01 = vfmaq_laneq_f16(cq01, aq01, bq01, 0);\
    cq02 = vfmaq_laneq_f16(cq02, aq01, bq01, 1);\
    cq03 = vfmaq_laneq_f16(cq03, aq01, bq01, 2);\
    cq04 = vfmaq_laneq_f16(cq04, aq01, bq01, 3);\
    aq01 = vld1q_f16(a_ptr);\
    cq05 = vfmaq_laneq_f16(cq05, aq02, bq01, 4);\
    cq06 = vfmaq_laneq_f16(cq06, aq02, bq01, 5);\
    cq07 = vfmaq_laneq_f16(cq07, aq02, bq01, 6);\
    cq08 = vfmaq_laneq_f16(cq08, aq02, bq01, 7);\
    aq02 = vld1q_f16(a_ptr + 8); a_ptr += 16;\
    bq01 = vld1q_f16(b_ptr1); b_ptr1 += 8;\
  }\
  if (k_left > 1) {\
    cq01 = vfmaq_laneq_f16(cq01, aq01, bq01, 0);\
    cq02 = vfmaq_laneq_f16(cq02, aq01, bq01, 1);\
    cq03 = vfmaq_laneq_f16(cq03, aq01, bq01, 2);\
    cq04 = vfmaq_laneq_f16(cq04, aq01, bq01, 3);\
    cq05 = vfmaq_laneq_f16(cq05, aq02, bq01, 4);\
    cq06 = vfmaq_laneq_f16(cq06, aq02, bq01, 5);\
    cq07 = vfmaq_laneq_f16(cq07, aq02, bq01, 6);\
    cq08 = vfmaq_laneq_f16(cq08, aq02, bq01, 7);\
    k_left -= 2;\
  }\
  cq01 = vaddq_f16(cq01, cq05);\
  cq02 = vaddq_f16(cq02, cq06);\
  cq03 = vaddq_f16(cq03, cq07);\
  cq04 = vaddq_f16(cq04, cq08);\
  if (k_left > 0) {\
    float16x4_t bd01 = vld1_f16(b_ptr1); b_ptr1 += 4;\
    aq01 = vld1q_f16(a_ptr); a_ptr += 8;\
    cq01 = vfmaq_lane_f16(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f16(cq02, aq01, bd01, 1);\
    cq03 = vfmaq_lane_f16(cq03, aq01, bd01, 2);\
    cq04 = vfmaq_lane_f16(cq04, aq01, bd01, 3);\
  }

#define KERNEL_M8N2_UNIT(a_head, b_head) \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x8_t cq01, cq02, cq03, cq04;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f16(0.0f);\
  float16x8_t aq01, aq02; float16x4_t bd01;\
  uint32_t k_left = K;\
  if (k_left > 1) {\
    aq01 = vld1q_f16(a_ptr);\
    aq02 = vld1q_f16(a_ptr + 8); a_ptr += 16;\
    bd01 = vld1_f16(b_ptr1); b_ptr1 += 4;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cq01 = vfmaq_lane_f16(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f16(cq02, aq01, bd01, 1);\
    aq01 = vld1q_f16(a_ptr);\
    cq03 = vfmaq_lane_f16(cq03, aq02, bd01, 2);\
    cq04 = vfmaq_lane_f16(cq04, aq02, bd01, 3);\
    aq02 = vld1q_f16(a_ptr + 8); a_ptr += 16;\
    bd01 = vld1_f16(b_ptr1); b_ptr1 += 4;\
  }\
  if (k_left > 1) {\
    cq01 = vfmaq_lane_f16(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f16(cq02, aq01, bd01, 1);\
    cq03 = vfmaq_lane_f16(cq03, aq02, bd01, 2);\
    cq04 = vfmaq_lane_f16(cq04, aq02, bd01, 3);\
    k_left -= 2;\
  }\
  cq01 = vaddq_f16(cq01, cq03);\
  cq02 = vaddq_f16(cq02, cq04);\
  if (k_left > 0) {\
    aq01 = vld1q_f16(a_ptr); a_ptr += 8;\
    float16_t bs1 = b_ptr1[0];\
    float16_t bs2 = b_ptr1[1]; b_ptr1 += 2;\
    cq01 = vfmaq_n_f16(cq01, aq01, bs1);\
    cq02 = vfmaq_n_f16(cq02, aq01, bs2);\
  }

#define KERNEL_M8N1_UNIT(a_head, b_head) \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x8_t cq01, cq02, cq03, cq04;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f16(0.0f);\
  float16x8_t aq01, aq02, aq03, aq04;\
  float16x4_t bd01;\
  uint32_t k_left = K;\
  if (k_left > 3) {\
    aq01 = vld1q_f16(a_ptr);\
    aq02 = vld1q_f16(a_ptr + 8);\
    aq03 = vld1q_f16(a_ptr + 16);\
    aq04 = vld1q_f16(a_ptr + 24); a_ptr += 32;\
    bd01 = vld1_f16(b_ptr1); b_ptr1 += 4;\
  }\
  for (; k_left > 7; k_left -= 4) {\
    cq01 = vfmaq_lane_f16(cq01, aq01, bd01, 0);\
    aq01 = vld1q_f16(a_ptr);\
    cq02 = vfmaq_lane_f16(cq02, aq02, bd01, 1);\
    aq02 = vld1q_f16(a_ptr + 8);\
    cq03 = vfmaq_lane_f16(cq03, aq03, bd01, 2);\
    aq03 = vld1q_f16(a_ptr + 16);\
    cq04 = vfmaq_lane_f16(cq04, aq04, bd01, 3);\
    aq04 = vld1q_f16(a_ptr + 24); a_ptr += 32;\
    bd01 = vld1_f16(b_ptr1); b_ptr1 += 4;\
  }\
  if (k_left > 3) {\
    cq01 = vfmaq_lane_f16(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f16(cq02, aq02, bd01, 1);\
    cq03 = vfmaq_lane_f16(cq03, aq03, bd01, 2);\
    cq04 = vfmaq_lane_f16(cq04, aq04, bd01, 3);\
    k_left -= 4;\
  }\
  cq01 = vaddq_f16(cq01, cq02);\
  cq03 = vaddq_f16(cq03, cq04);\
  cq01 = vaddq_f16(cq01, cq03);\
  for (; k_left > 0; k_left--) {\
    aq01 = vld1q_f16(a_ptr); a_ptr += 8;\
    float16_t bs1 = *b_ptr1; b_ptr1++;\
    cq01 = vfmaq_n_f16(cq01, aq01, bs1);\
  }

#define KERNEL_M8N4 KERNEL_M8N4_UNIT(a_head, b_head)
#define KERNEL_M8N2 KERNEL_M8N2_UNIT(a_head, b_head)
#define KERNEL_M8N1 KERNEL_M8N1_UNIT(a_head, b_head)
#define KERNEL_M4N8 KERNEL_M8N4_UNIT(b_head, a_head)
#define KERNEL_M2N8 KERNEL_M8N2_UNIT(b_head, a_head)
#define KERNEL_M1N8 KERNEL_M8N1_UNIT(b_head, a_head)

#define KERNEL_M4N16_UNIT(a_head, b_head) \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x8_t cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f16(0.0f);\
  cq05 = cq06 = cq07 = cq08 = vdupq_n_f16(0.0f);\
  float16x8_t aq01, bq01, bq02, bq03, bq04;\
  uint32_t k_left = K;\
  if (k_left > 1) {\
    aq01 = vld1q_f16(a_ptr); a_ptr += 8;\
    bq01 = vld1q_f16(b_ptr1);\
    bq02 = vld1q_f16(b_ptr1 + 8);\
    bq03 = vld1q_f16(b_ptr1 + 16);\
    bq04 = vld1q_f16(b_ptr1 + 24); b_ptr1 += 32;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cq01 = vfmaq_laneq_f16(cq01, bq01, aq01, 0);\
    cq03 = vfmaq_laneq_f16(cq03, bq01, aq01, 1);\
    cq05 = vfmaq_laneq_f16(cq05, bq01, aq01, 2);\
    cq07 = vfmaq_laneq_f16(cq07, bq01, aq01, 3);\
    bq01 = vld1q_f16(b_ptr1);\
    cq02 = vfmaq_laneq_f16(cq02, bq02, aq01, 0);\
    cq04 = vfmaq_laneq_f16(cq04, bq02, aq01, 1);\
    cq06 = vfmaq_laneq_f16(cq06, bq02, aq01, 2);\
    cq08 = vfmaq_laneq_f16(cq08, bq02, aq01, 3);\
    bq02 = vld1q_f16(b_ptr1 + 8);\
    cq01 = vfmaq_laneq_f16(cq01, bq03, aq01, 4);\
    cq03 = vfmaq_laneq_f16(cq03, bq03, aq01, 5);\
    cq05 = vfmaq_laneq_f16(cq05, bq03, aq01, 6);\
    cq07 = vfmaq_laneq_f16(cq07, bq03, aq01, 7);\
    bq03 = vld1q_f16(b_ptr1 + 16);\
    cq02 = vfmaq_laneq_f16(cq02, bq04, aq01, 4);\
    cq04 = vfmaq_laneq_f16(cq04, bq04, aq01, 5);\
    cq06 = vfmaq_laneq_f16(cq06, bq04, aq01, 6);\
    cq08 = vfmaq_laneq_f16(cq08, bq04, aq01, 7);\
    bq04 = vld1q_f16(b_ptr1 + 24); b_ptr1 += 32;\
    aq01 = vld1q_f16(a_ptr); a_ptr += 8;\
  }\
  if (k_left > 1) {\
    cq01 = vfmaq_laneq_f16(cq01, bq01, aq01, 0);\
    cq03 = vfmaq_laneq_f16(cq03, bq01, aq01, 1);\
    cq05 = vfmaq_laneq_f16(cq05, bq01, aq01, 2);\
    cq07 = vfmaq_laneq_f16(cq07, bq01, aq01, 3);\
    cq02 = vfmaq_laneq_f16(cq02, bq02, aq01, 0);\
    cq04 = vfmaq_laneq_f16(cq04, bq02, aq01, 1);\
    cq06 = vfmaq_laneq_f16(cq06, bq02, aq01, 2);\
    cq08 = vfmaq_laneq_f16(cq08, bq02, aq01, 3);\
    cq01 = vfmaq_laneq_f16(cq01, bq03, aq01, 4);\
    cq03 = vfmaq_laneq_f16(cq03, bq03, aq01, 5);\
    cq05 = vfmaq_laneq_f16(cq05, bq03, aq01, 6);\
    cq07 = vfmaq_laneq_f16(cq07, bq03, aq01, 7);\
    cq02 = vfmaq_laneq_f16(cq02, bq04, aq01, 4);\
    cq04 = vfmaq_laneq_f16(cq04, bq04, aq01, 5);\
    cq06 = vfmaq_laneq_f16(cq06, bq04, aq01, 6);\
    cq08 = vfmaq_laneq_f16(cq08, bq04, aq01, 7);\
    k_left -= 2;\
  }\
  if (k_left > 0) {\
    float16x4_t ad01 = vld1_f16(a_ptr); a_ptr += 4;\
    bq01 = vld1q_f16(b_ptr1);\
    bq02 = vld1q_f16(b_ptr1 + 8); b_ptr1 += 16;\
    cq01 = vfmaq_lane_f16(cq01, bq01, ad01, 0);\
    cq03 = vfmaq_lane_f16(cq03, bq01, ad01, 1);\
    cq05 = vfmaq_lane_f16(cq05, bq01, ad01, 2);\
    cq07 = vfmaq_lane_f16(cq07, bq01, ad01, 3);\
    cq02 = vfmaq_lane_f16(cq02, bq02, ad01, 0);\
    cq04 = vfmaq_lane_f16(cq04, bq02, ad01, 1);\
    cq06 = vfmaq_lane_f16(cq06, bq02, ad01, 2);\
    cq08 = vfmaq_lane_f16(cq08, bq02, ad01, 3);\
  }

#define KERNEL_M2N16_UNIT(a_head, b_head) \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x8_t cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f16(0.0f);\
  cq05 = cq06 = cq07 = cq08 = vdupq_n_f16(0.0f);\
  float16x8_t bq01, bq02, bq03, bq04;\
  float16x4_t ad01;\
  uint32_t k_left = K;\
  if (k_left > 1) {\
    ad01 = vld1_f16(a_ptr); a_ptr += 4;\
    bq01 = vld1q_f16(b_ptr1);\
    bq02 = vld1q_f16(b_ptr1 + 8);\
    bq03 = vld1q_f16(b_ptr1 + 16);\
    bq04 = vld1q_f16(b_ptr1 + 24); b_ptr1 += 32;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cq01 = vfmaq_lane_f16(cq01, bq01, ad01, 0);\
    cq03 = vfmaq_lane_f16(cq03, bq01, ad01, 1);\
    bq01 = vld1q_f16(b_ptr1);\
    cq02 = vfmaq_lane_f16(cq02, bq02, ad01, 0);\
    cq04 = vfmaq_lane_f16(cq04, bq02, ad01, 1);\
    bq02 = vld1q_f16(b_ptr1 + 8);\
    cq05 = vfmaq_lane_f16(cq05, bq03, ad01, 2);\
    cq07 = vfmaq_lane_f16(cq07, bq03, ad01, 3);\
    bq03 = vld1q_f16(b_ptr1 + 16);\
    cq06 = vfmaq_lane_f16(cq06, bq04, ad01, 2);\
    cq08 = vfmaq_lane_f16(cq08, bq04, ad01, 3);\
    bq04 = vld1q_f16(b_ptr1 + 24); b_ptr1 += 32;\
    ad01 = vld1_f16(a_ptr); a_ptr += 4;\
  }\
  if (k_left > 1) {\
    cq01 = vfmaq_lane_f16(cq01, bq01, ad01, 0);\
    cq03 = vfmaq_lane_f16(cq03, bq01, ad01, 1);\
    cq05 = vfmaq_lane_f16(cq05, bq03, ad01, 2);\
    cq07 = vfmaq_lane_f16(cq07, bq03, ad01, 3);\
    cq02 = vfmaq_lane_f16(cq02, bq02, ad01, 0);\
    cq04 = vfmaq_lane_f16(cq04, bq02, ad01, 1);\
    cq06 = vfmaq_lane_f16(cq06, bq04, ad01, 2);\
    cq08 = vfmaq_lane_f16(cq08, bq04, ad01, 3);\
    k_left -= 2;\
  }\
  cq01 = vaddq_f16(cq01, cq05);\
  cq02 = vaddq_f16(cq02, cq06);\
  cq03 = vaddq_f16(cq03, cq07);\
  cq04 = vaddq_f16(cq04, cq08);\
  if (k_left > 0) {\
    bq01 = vld1q_f16(b_ptr1);\
    bq02 = vld1q_f16(b_ptr1 + 8); b_ptr1 += 16;\
    float16_t as1 = a_ptr[0];\
    float16_t as2 = a_ptr[1]; a_ptr += 2;\
    cq01 = vfmaq_n_f16(cq01, bq01, as1);\
    cq02 = vfmaq_n_f16(cq02, bq02, as1);\
    cq03 = vfmaq_n_f16(cq03, bq01, as2);\
    cq04 = vfmaq_n_f16(cq04, bq02, as2);\
  }

#define KERNEL_M1N16_UNIT(a_head, b_head) \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x8_t cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f16(0.0f);\
  cq05 = cq06 = cq07 = cq08 = vdupq_n_f16(0.0f);\
  float16x8_t bq01, bq02, bq03, bq04, bq05, bq06, bq07, bq08;\
  float16x4_t ad01;\
  uint32_t k_left = K;\
  if (k_left > 3) {\
    ad01 = vld1_f16(a_ptr); a_ptr += 4;\
    bq01 = vld1q_f16(b_ptr1);\
    bq02 = vld1q_f16(b_ptr1 + 8);\
    bq03 = vld1q_f16(b_ptr1 + 16);\
    bq04 = vld1q_f16(b_ptr1 + 24);\
    bq05 = vld1q_f16(b_ptr1 + 32);\
    bq06 = vld1q_f16(b_ptr1 + 40);\
    bq07 = vld1q_f16(b_ptr1 + 48);\
    bq08 = vld1q_f16(b_ptr1 + 56); b_ptr1 += 64;\
  }\
  for (; k_left > 7; k_left -= 4) {\
    cq01 = vfmaq_lane_f16(cq01, bq01, ad01, 0);\
    bq01 = vld1q_f16(b_ptr1);\
    cq02 = vfmaq_lane_f16(cq02, bq02, ad01, 0);\
    bq02 = vld1q_f16(b_ptr1 + 8);\
    cq03 = vfmaq_lane_f16(cq03, bq03, ad01, 1);\
    bq03 = vld1q_f16(b_ptr1 + 16);\
    cq04 = vfmaq_lane_f16(cq04, bq04, ad01, 1);\
    bq04 = vld1q_f16(b_ptr1 + 24);\
    cq05 = vfmaq_lane_f16(cq05, bq05, ad01, 2);\
    bq05 = vld1q_f16(b_ptr1 + 32);\
    cq06 = vfmaq_lane_f16(cq06, bq06, ad01, 2);\
    bq06 = vld1q_f16(b_ptr1 + 40);\
    cq07 = vfmaq_lane_f16(cq07, bq07, ad01, 3);\
    bq07 = vld1q_f16(b_ptr1 + 48);\
    cq08 = vfmaq_lane_f16(cq08, bq08, ad01, 3);\
    bq08 = vld1q_f16(b_ptr1 + 56); b_ptr1 += 64;\
    ad01 = vld1_f16(a_ptr); a_ptr += 4;\
  }\
  if (k_left > 3) {\
    cq01 = vfmaq_lane_f16(cq01, bq01, ad01, 0);\
    cq03 = vfmaq_lane_f16(cq03, bq03, ad01, 1);\
    cq05 = vfmaq_lane_f16(cq05, bq05, ad01, 2);\
    cq07 = vfmaq_lane_f16(cq07, bq07, ad01, 3);\
    cq02 = vfmaq_lane_f16(cq02, bq02, ad01, 0);\
    cq04 = vfmaq_lane_f16(cq04, bq04, ad01, 1);\
    cq06 = vfmaq_lane_f16(cq06, bq06, ad01, 2);\
    cq08 = vfmaq_lane_f16(cq08, bq08, ad01, 3);\
    k_left -= 4;\
  }\
  cq01 = vaddq_f16(cq01, cq03);\
  cq05 = vaddq_f16(cq05, cq07);\
  cq02 = vaddq_f16(cq02, cq04);\
  cq06 = vaddq_f16(cq06, cq08);\
  cq01 = vaddq_f16(cq01, cq05);\
  cq02 = vaddq_f16(cq02, cq06);\
  for (; k_left > 0; k_left--) {\
    float16_t as1 = *a_ptr; a_ptr++;\
    bq01 = vld1q_f16(b_ptr1);\
    bq02 = vld1q_f16(b_ptr1 + 8); b_ptr1 += 16;\
    cq01 = vfmaq_n_f16(cq01, bq01, as1);\
    cq02 = vfmaq_n_f16(cq02, bq02, as1);\
  }

#define KERNEL_M4N16 KERNEL_M4N16_UNIT(a_head, b_head)
#define KERNEL_M2N16 KERNEL_M2N16_UNIT(a_head, b_head)
#define KERNEL_M1N16 KERNEL_M1N16_UNIT(a_head, b_head)
#define KERNEL_M16N4 KERNEL_M4N16_UNIT(b_head, a_head)
#define KERNEL_M16N2 KERNEL_M2N16_UNIT(b_head, a_head)
#define KERNEL_M16N1 KERNEL_M1N16_UNIT(b_head, a_head)

#define KERNEL_M4N4 \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x4_t cd01, cd02, cd03, cd04;\
  cd01 = cd02 = cd03 = cd04 = vdup_n_f16(0.0f);\
  float16x4_t ad01, bd01;\
  uint32_t k_left = K;\
  for (; k_left > 0; k_left--) {\
    ad01 = vld1_f16(a_ptr); a_ptr += 4;\
    bd01 = vld1_f16(b_ptr1); b_ptr1 += 4;\
    cd01 = vfma_lane_f16(cd01, ad01, bd01, 0);\
    cd02 = vfma_lane_f16(cd02, ad01, bd01, 1);\
    cd03 = vfma_lane_f16(cd03, ad01, bd01, 2);\
    cd04 = vfma_lane_f16(cd04, ad01, bd01, 3);\
  }

#define KERNEL_M4N2_UNIT(a_head, b_head) \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x4_t cd01, cd02, cd03, cd04;\
  cd01 = cd02 = cd03 = cd04 = vdup_n_f16(0.0f);\
  float16x4_t ad01, ad02, bd01;\
  uint32_t k_left = K;\
  for (; k_left > 1; k_left -= 2) {\
    ad01 = vld1_f16(a_ptr);\
    ad02 = vld1_f16(a_ptr + 4); a_ptr += 8;\
    bd01 = vld1_f16(b_ptr1); b_ptr1 += 4;\
    cd01 = vfma_lane_f16(cd01, ad01, bd01, 0);\
    cd02 = vfma_lane_f16(cd02, ad01, bd01, 1);\
    cd03 = vfma_lane_f16(cd03, ad02, bd01, 2);\
    cd04 = vfma_lane_f16(cd04, ad02, bd01, 3);\
  }\
  cd01 = vadd_f16(cd01, cd03);\
  cd02 = vadd_f16(cd02, cd04);\
  if (k_left > 0) {\
    ad01 = vld1_f16(a_ptr); a_ptr += 4;\
    float16_t bs1 = b_ptr1[0];\
    float16_t bs2 = b_ptr1[1]; b_ptr1 += 2;\
    cd01 = vfma_n_f16(cd01, ad01, bs1);\
    cd02 = vfma_n_f16(cd02, ad01, bs2);\
  }

#define KERNEL_M4N1_UNIT(a_head, b_head) \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x4_t cd01, cd02, cd03, cd04;\
  cd01 = cd02 = cd03 = cd04 = vdup_n_f16(0.0f);\
  float16x4_t ad01, ad02, ad03, ad04, bd01;\
  uint32_t k_left = K;\
  for (; k_left > 3; k_left -= 4) {\
    ad01 = vld1_f16(a_ptr);\
    ad02 = vld1_f16(a_ptr + 4);\
    ad03 = vld1_f16(a_ptr + 8);\
    ad04 = vld1_f16(a_ptr + 12); a_ptr += 16;\
    bd01 = vld1_f16(b_ptr1); b_ptr1 += 4;\
    cd01 = vfma_lane_f16(cd01, ad01, bd01, 0);\
    cd02 = vfma_lane_f16(cd02, ad02, bd01, 1);\
    cd03 = vfma_lane_f16(cd03, ad03, bd01, 2);\
    cd04 = vfma_lane_f16(cd04, ad04, bd01, 3);\
  }\
  cd01 = vadd_f16(cd01, cd03);\
  cd02 = vadd_f16(cd02, cd04);\
  cd01 = vadd_f16(cd01, cd02);\
  for (; k_left > 0; k_left--) {\
    ad01 = vld1_f16(a_ptr); a_ptr += 4;\
    float16_t bs1 = *b_ptr1; b_ptr1++;\
    cd01 = vfma_n_f16(cd01, ad01, bs1);\
  }

#define KERNEL_M4N2 KERNEL_M4N2_UNIT(a_head, b_head)
#define KERNEL_M4N1 KERNEL_M4N1_UNIT(a_head, b_head)
#define KERNEL_M2N4 KERNEL_M4N2_UNIT(b_head, a_head)
#define KERNEL_M1N4 KERNEL_M4N1_UNIT(b_head, a_head)

#define KERNEL_M2N2 \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16_t cs1, cs2, cs3, cs4;\
  cs1 = cs2 = cs3 = cs4 = 0.0f;\
  float16_t as1, as2, bs1, bs2;\
  uint32_t k_left = K;\
  for (; k_left > 0; k_left--) {\
    as1 = a_ptr[0]; as2 = a_ptr[1]; a_ptr += 2;\
    bs1 = b_ptr1[0]; bs2 = b_ptr1[1]; b_ptr1 += 2;\
    cs1 += as1 * bs1;\
    cs2 += as2 * bs1;\
    cs3 += as1 * bs2;\
    cs4 += as2 * bs2;\
  }

#define KERNEL_M2N1_UNIT(a_head, b_head) \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16_t cs1, cs2; cs1 = cs2 = 0.0f;\
  float16_t as1, as2, bs1;\
  uint32_t k_left = K;\
  for (; k_left > 0; k_left--) {\
    as1 = a_ptr[0]; as2 = a_ptr[1]; a_ptr += 2;\
    bs1 = b_ptr1[0]; b_ptr1++;\
    cs1 += as1 * bs1;\
    cs2 += as2 * bs1;\
  }

#define KERNEL_M2N1 KERNEL_M2N1_UNIT(a_head, b_head)
#define KERNEL_M1N2 KERNEL_M2N1_UNIT(b_head, a_head)

#define KERNEL_M1N1 \
  const float16_t *a_ptr = a_head;\
  const float16_t *b_ptr1 = b_head;\
  float16x4_t cd01 = vdup_n_f16(0.0f);\
  float16x4_t ad01, bd01;\
  uint32_t k_left = K;\
  for (; k_left > 3; k_left -= 4) {\
    ad01 = vld1_f16(a_ptr); a_ptr += 4;\
    bd01 = vld1_f16(b_ptr1); b_ptr1 += 4;\
    cd01 = vfma_f16(cd01, ad01, bd01);\
  }\
  float16_t cs1 = vget_lane_f16(cd01, 0) + vget_lane_f16(cd01, 1) + \
    vget_lane_f16(cd01, 2) + vget_lane_f16(cd01, 3);\
  for (; k_left > 0; k_left--) {\
    cs1 += (*a_ptr) * (*b_ptr1); a_ptr++; b_ptr1++;\
  }


#define SAVE_M1N8_UNIT(cq01, c_tmp) {\
  float16_t cs1 = vgetq_lane_f16(cq01, 0);\
  float16_t cs2 = vgetq_lane_f16(cq01, 1);\
  float16_t cs3 = vgetq_lane_f16(cq01, 2);\
  float16_t cs4 = vgetq_lane_f16(cq01, 3);\
  float16_t cs5 = vgetq_lane_f16(cq01, 4);\
  float16_t cs6 = vgetq_lane_f16(cq01, 5);\
  float16_t cs7 = vgetq_lane_f16(cq01, 6);\
  float16_t cs8 = vgetq_lane_f16(cq01, 7);\
  *c_tmp = *c_tmp * beta + cs1; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs2; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs3; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs4; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs5; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs6; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs7; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs8; c_tmp += ldc;\
}

#define SAVE_M2N8_UNIT(cq01, cq02, c_tmp) {\
  float16x8x2_t cqd1;\
  cqd1.val[0] = vdupq_n_f16(0.0f);\
  cqd1.val[1] = vdupq_n_f16(0.0f);\
  cqd1 = vld2q_lane_f16(c_tmp, cqd1, 0); c_tmp += ldc;\
  cqd1 = vld2q_lane_f16(c_tmp, cqd1, 1); c_tmp += ldc;\
  cqd1 = vld2q_lane_f16(c_tmp, cqd1, 2); c_tmp += ldc;\
  cqd1 = vld2q_lane_f16(c_tmp, cqd1, 3); c_tmp += ldc;\
  cqd1 = vld2q_lane_f16(c_tmp, cqd1, 4); c_tmp += ldc;\
  cqd1 = vld2q_lane_f16(c_tmp, cqd1, 5); c_tmp += ldc;\
  cqd1 = vld2q_lane_f16(c_tmp, cqd1, 6); c_tmp += ldc;\
  cqd1 = vld2q_lane_f16(c_tmp, cqd1, 7); c_tmp -= ldc * 7;\
  cqd1.val[0] = vfmaq_n_f16(cq01, cqd1.val[0], beta);\
  cqd1.val[1] = vfmaq_n_f16(cq02, cqd1.val[1], beta);\
  vst2q_lane_f16(c_tmp, cqd1, 0); c_tmp += ldc;\
  vst2q_lane_f16(c_tmp, cqd1, 1); c_tmp += ldc;\
  vst2q_lane_f16(c_tmp, cqd1, 2); c_tmp += ldc;\
  vst2q_lane_f16(c_tmp, cqd1, 3); c_tmp += ldc;\
  vst2q_lane_f16(c_tmp, cqd1, 4); c_tmp += ldc;\
  vst2q_lane_f16(c_tmp, cqd1, 5); c_tmp += ldc;\
  vst2q_lane_f16(c_tmp, cqd1, 6); c_tmp += ldc;\
  vst2q_lane_f16(c_tmp, cqd1, 7); c_tmp += ldc;\
}

#define SAVE_M4N8_UNIT(cq01, cq02, cq03, cq04, c_tmp) {\
  float16x8x4_t cqq1;\
  cqq1.val[0] = vdupq_n_f16(0.0f);\
  cqq1.val[1] = vdupq_n_f16(0.0f);\
  cqq1.val[2] = vdupq_n_f16(0.0f);\
  cqq1.val[3] = vdupq_n_f16(0.0f);\
  cqq1 = vld4q_lane_f16(c_tmp, cqq1, 0); c_tmp += ldc;\
  cqq1 = vld4q_lane_f16(c_tmp, cqq1, 1); c_tmp += ldc;\
  cqq1 = vld4q_lane_f16(c_tmp, cqq1, 2); c_tmp += ldc;\
  cqq1 = vld4q_lane_f16(c_tmp, cqq1, 3); c_tmp += ldc;\
  cqq1 = vld4q_lane_f16(c_tmp, cqq1, 4); c_tmp += ldc;\
  cqq1 = vld4q_lane_f16(c_tmp, cqq1, 5); c_tmp += ldc;\
  cqq1 = vld4q_lane_f16(c_tmp, cqq1, 6); c_tmp += ldc;\
  cqq1 = vld4q_lane_f16(c_tmp, cqq1, 7); c_tmp -= ldc * 7;\
  cqq1.val[0] = vfmaq_n_f16(cq01, cqq1.val[0], beta);\
  cqq1.val[1] = vfmaq_n_f16(cq02, cqq1.val[1], beta);\
  cqq1.val[2] = vfmaq_n_f16(cq03, cqq1.val[2], beta);\
  cqq1.val[3] = vfmaq_n_f16(cq04, cqq1.val[3], beta);\
  vst4q_lane_f16(c_tmp, cqq1, 0); c_tmp += ldc;\
  vst4q_lane_f16(c_tmp, cqq1, 1); c_tmp += ldc;\
  vst4q_lane_f16(c_tmp, cqq1, 2); c_tmp += ldc;\
  vst4q_lane_f16(c_tmp, cqq1, 3); c_tmp += ldc;\
  vst4q_lane_f16(c_tmp, cqq1, 4); c_tmp += ldc;\
  vst4q_lane_f16(c_tmp, cqq1, 5); c_tmp += ldc;\
  vst4q_lane_f16(c_tmp, cqq1, 6); c_tmp += ldc;\
  vst4q_lane_f16(c_tmp, cqq1, 7); c_tmp += ldc;\
}

#define SAVE_M2N4_UNIT(cd01, cd02, c_tmp) {\
  float16x4x2_t cdd1;\
  cdd1.val[0] = vdup_n_f16(0.0f);\
  cdd1.val[1] = vdup_n_f16(0.0f);\
  cdd1 = vld2_lane_f16(c_tmp, cdd1, 0); c_tmp += ldc;\
  cdd1 = vld2_lane_f16(c_tmp, cdd1, 1); c_tmp += ldc;\
  cdd1 = vld2_lane_f16(c_tmp, cdd1, 2); c_tmp += ldc;\
  cdd1 = vld2_lane_f16(c_tmp, cdd1, 3); c_tmp -= ldc * 3;\
  cdd1.val[0] = vfma_n_f16(cd01, cdd1.val[0], beta);\
  cdd1.val[1] = vfma_n_f16(cd02, cdd1.val[1], beta);\
  vst2_lane_f16(c_tmp, cdd1, 0); c_tmp += ldc;\
  vst2_lane_f16(c_tmp, cdd1, 1); c_tmp += ldc;\
  vst2_lane_f16(c_tmp, cdd1, 2); c_tmp += ldc;\
  vst2_lane_f16(c_tmp, cdd1, 3); c_tmp += ldc;\
}

#define SAVE_M1N4_UNIT(cd01, c_tmp) {\
  float16_t cs1 = vget_lane_f16(cd01, 0);\
  float16_t cs2 = vget_lane_f16(cd01, 1);\
  float16_t cs3 = vget_lane_f16(cd01, 2);\
  float16_t cs4 = vget_lane_f16(cd01, 3);\
  *c_tmp = *c_tmp * beta + cs1; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs2; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs3; c_tmp += ldc;\
  *c_tmp = *c_tmp * beta + cs4; c_tmp += ldc;\
}

#define SAVE_M16N2_UNIT(cq01, cq02, cq03, cq04, c_tmp) \
  cq01 = vfmaq_n_f16(cq01, vld1q_f16(c_tmp), beta);\
  cq02 = vfmaq_n_f16(cq02, vld1q_f16(c_tmp + 8), beta);\
  cq03 = vfmaq_n_f16(cq03, vld1q_f16(c_tmp + ldc), beta);\
  cq04 = vfmaq_n_f16(cq04, vld1q_f16(c_tmp + ldc + 8), beta);\
  vst1q_f16(c_tmp, cq01); vst1q_f16(c_tmp + 8, cq02);\
  vst1q_f16(c_tmp + ldc, cq03); vst1q_f16(c_tmp + ldc + 8, cq04);\
  c_tmp += ldc * 2;

#define SAVE_M8N2_UNIT(cq01, cq02, c_tmp) \
  cq01 = vfmaq_n_f16(cq01, vld1q_f16(c_tmp), beta);\
  cq02 = vfmaq_n_f16(cq02, vld1q_f16(c_tmp + ldc), beta);\
  vst1q_f16(c_tmp, cq01);\
  vst1q_f16(c_tmp + ldc, cq02); c_tmp += ldc * 2;

#define SAVE_M4N2_UNIT(cd01, cd02, c_tmp) \
  cd01 = vfma_n_f16(cd01, vld1_f16(c_tmp), beta);\
  cd02 = vfma_n_f16(cd02, vld1_f16(c_tmp + ldc), beta);\
  vst1_f16(c_tmp, cd01);\
  vst1_f16(c_tmp + ldc, cd02); c_tmp += ldc * 2;

#define SAVE_M8N16 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M8N2_UNIT(cq01, cq02, c_tmp)\
  SAVE_M8N2_UNIT(cq03, cq04, c_tmp)\
  SAVE_M8N2_UNIT(cq05, cq06, c_tmp)\
  SAVE_M8N2_UNIT(cq07, cq08, c_tmp)\
  SAVE_M8N2_UNIT(cq09, cq10, c_tmp)\
  SAVE_M8N2_UNIT(cq11, cq12, c_tmp)\
  SAVE_M8N2_UNIT(cq13, cq14, c_tmp)\
  SAVE_M8N2_UNIT(cq15, cq16, c_tmp)

#define SAVE_M4N16 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M4N8_UNIT(cq01, cq03, cq05, cq07, c_tmp)\
  SAVE_M4N8_UNIT(cq02, cq04, cq06, cq08, c_tmp)

#define SAVE_M2N16 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M2N8_UNIT(cq01, cq03, c_tmp)\
  SAVE_M2N8_UNIT(cq02, cq04, c_tmp)

#define SAVE_M1N16 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M1N8_UNIT(cq01, c_tmp)\
  SAVE_M1N8_UNIT(cq02, c_tmp)

#define SAVE_M16N8 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M16N2_UNIT(cq01, cq02, cq03, cq04, c_tmp)\
  SAVE_M16N2_UNIT(cq05, cq06, cq07, cq08, c_tmp)\
  SAVE_M16N2_UNIT(cq09, cq10, cq11, cq12, c_tmp)\
  SAVE_M16N2_UNIT(cq13, cq14, cq15, cq16, c_tmp)

#define SAVE_M8N8 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M8N2_UNIT(cq01, cq02, c_tmp)\
  SAVE_M8N2_UNIT(cq03, cq04, c_tmp)\
  SAVE_M8N2_UNIT(cq05, cq06, c_tmp)\
  SAVE_M8N2_UNIT(cq07, cq08, c_tmp)

#define SAVE_M4N8 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M4N8_UNIT(cq01, cq02, cq03, cq04, c_tmp)

#define SAVE_M2N8 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M2N8_UNIT(cq01, cq02, c_tmp)

#define SAVE_M1N8 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M1N8_UNIT(cq01, c_tmp)

#define SAVE_M16N4 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M16N2_UNIT(cq01, cq02, cq03, cq04, c_tmp)\
  SAVE_M16N2_UNIT(cq05, cq06, cq07, cq08, c_tmp)

#define SAVE_M8N4 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M8N2_UNIT(cq01, cq02, c_tmp)\
  SAVE_M8N2_UNIT(cq03, cq04, c_tmp)

#define SAVE_M4N4 \
  float16_t *c_tmp = c_ptr;\
  SAVE_M4N2_UNIT(cd01, cd02, c_tmp)\
  SAVE_M4N2_UNIT(cd03, cd04, c_tmp)

#define SAVE_M2N4 \
  float16_t *c_tmp = c_ptr; SAVE_M2N4_UNIT(cd01, cd02, c_tmp)

#define SAVE_M1N4 \
  float16_t *c_tmp = c_ptr; SAVE_M1N4_UNIT(cd01, c_tmp)

#define SAVE_M16N2 \
  float16_t *c_tmp = c_ptr; SAVE_M16N2_UNIT(cq01, cq02, cq03, cq04, c_tmp)

#define SAVE_M8N2 \
  float16_t *c_tmp = c_ptr; SAVE_M8N2_UNIT(cq01, cq02, c_tmp)

#define SAVE_M4N2 \
  float16_t *c_tmp = c_ptr; SAVE_M4N2_UNIT(cd01, cd02, c_tmp)

#define SAVE_M2N2 \
  c_ptr[0] = c_ptr[0] * beta + cs1;\
  c_ptr[1] = c_ptr[1] * beta + cs2;\
  c_ptr[ldc] = c_ptr[ldc] * beta + cs3;\
  c_ptr[ldc + 1] = c_ptr[ldc + 1] * beta + cs4;\

#define SAVE_M1N2 \
  c_ptr[0] = c_ptr[0] * beta + cs1;\
  c_ptr[ldc] = c_ptr[ldc] * beta + cs2;\

#define SAVE_M16N1 \
  cq01 = vfmaq_n_f16(cq01, vld1q_f16(c_ptr), beta);\
  cq02 = vfmaq_n_f16(cq02, vld1q_f16(c_ptr + 8), beta);\
  vst1q_f16(c_ptr, cq01); vst1q_f16(c_ptr + 8, cq02);

#define SAVE_M8N1 \
  cq01 = vfmaq_n_f16(cq01, vld1q_f16(c_ptr), beta);\
  vst1q_f16(c_ptr, cq01);

#define SAVE_M4N1 \
  cd01 = vfma_n_f16(cd01, vld1_f16(c_ptr), beta);\
  vst1_f16(c_ptr, cd01);

#define SAVE_M2N1 \
  c_ptr[0] = c_ptr[0] * beta + cs1;\
  c_ptr[1] = c_ptr[1] * beta + cs2;\

#define SAVE_M1N1 \
  c_ptr[0] = c_ptr[0] * beta + cs1;

#define NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(mdim, ndim) \
static inline void\
  inline_dualpack_gemm_afloat16_t_bfloat16_t_cfloat16_t_m##mdim##_n##ndim(\
  const float16_t *a_head, const float16_t *b_head, float16_t *c_ptr,\
  uint32_t K, float16_t beta, uint32_t ldc) {\
  KERNEL_M##mdim##N##ndim\
  SAVE_M##mdim##N##ndim\
}

NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 1)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 2)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 1)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 2)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 4)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 4)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 1)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 2)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 4)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 8)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 8)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 8)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 1)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 2)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 4)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 16)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 16)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 16)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(16, 1)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(16, 2)
NEON_HGEMM_INLINE_DUALPACK_UNIT_FUNC(16, 4)

#define CPUID_DETECT_MNK 1000000

void hgemm_kernel_lm_m8n16(uint32_t M, uint32_t N, uint32_t K, float16_t beta,
  const float16_t * __restrict__ sa, const float16_t * __restrict__ sb,
  float16_t * __restrict__ C, uint32_t ldc) {

  uint32_t n_left = N;
  const float16_t *b_head = sb;
  float16_t *c_head = C;
  uint32_t acc_mnk = CPUID_DETECT_MNK;
  uint8_t cpuid = 0, cputype = 0;

  for (; n_left > 15; n_left -= 16) {
    if (acc_mnk >= CPUID_DETECT_MNK) {
      cpuid = sched_getcpu();
      cputype = blas_arm_get_cpu_type(cpuid);
      acc_mnk = 0;
    }
    const float16_t *a_head = sa;
    float16_t *c_ptr = c_head;
    uint32_t m_left = M;
    if (cputype == 55) {
      for (; m_left > 7; m_left -= 8) {
        KERNEL_M8N16_A55
        SAVE_M8N16
        a_head += 8 * K;
        c_ptr += 8;
      }
    } else {
      for (; m_left > 7; m_left -= 8) {
        KERNEL_M8N16_A76
        SAVE_M8N16
        a_head += 8 * K;
        c_ptr += 8;
      }
    }
    MICRO_COMPUTE_LM(4, 16, float16_t, float16_t, float16_t)
    b_head += K * 16;
    c_head += ldc * 16;
    acc_mnk += 16 * K * M;
  }

  for (; n_left > 7; n_left -= 8) {
    if (acc_mnk >= CPUID_DETECT_MNK) {
      cpuid = sched_getcpu();
      cputype = blas_arm_get_cpu_type(cpuid);
      acc_mnk = 0;
    }
    const float16_t *a_head = sa;
    float16_t *c_ptr = c_head;
    uint32_t m_left = M;
    if (cputype == 55) {
      for (; m_left > 7; m_left -= 8) {
        KERNEL_M8N8_A55
        SAVE_M8N8
        a_head += 8 * K;
        c_ptr += 8;
      }
    } else {
      for (; m_left > 7; m_left -= 8) {
        KERNEL_M8N8_A76
        SAVE_M8N8
        a_head += 8 * K;
        c_ptr += 8;
      }
    }
    MICRO_COMPUTE_LM(4, 8, float16_t, float16_t, float16_t)
    b_head += K * 8;
    c_head += ldc * 8;
    acc_mnk += 8 * K * M;
  }

  ASSEMBLE_DUALPACK_COMPUTE_LM(4, float16_t, float16_t, float16_t, 8)
}

void hgemm_kernel_ln_m16n8(uint32_t M, uint32_t N, uint32_t K, float16_t beta,
  const float16_t * __restrict__ sa, const float16_t * __restrict__ sb,
  float16_t * __restrict__ C, uint32_t ldc) {

  uint32_t m_left = M;
  const float16_t *a_head = sa;
  float16_t *c_head = C;
  uint32_t acc_mnk = CPUID_DETECT_MNK;
  uint8_t cpuid = 0, cputype = 0;
  for (; m_left > 15; m_left -= 16) {
    if (acc_mnk >= CPUID_DETECT_MNK) {
      cpuid = sched_getcpu();
      cputype = blas_arm_get_cpu_type(cpuid);
      acc_mnk = 0;
    }
    const float16_t *b_head = sb;
    float16_t *c_ptr = c_head;
    uint32_t n_left = N;
    if (cputype == 55) {
      for (; n_left > 7; n_left -= 8) {
        KERNEL_M16N8_A55
        SAVE_M16N8
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    } else {
      for (; n_left > 7; n_left -= 8) {
        KERNEL_M16N8_A76
        SAVE_M16N8
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    }
    MICRO_COMPUTE_LN(16, 4, float16_t, float16_t, float16_t)
    a_head += K * 16;
    c_head += 16;
    acc_mnk += 16 * N * K;
  }

  for (; m_left > 7; m_left -= 8) {
    if (acc_mnk >= CPUID_DETECT_MNK) {
      cpuid = sched_getcpu();
      cputype = blas_arm_get_cpu_type(cpuid);
      acc_mnk = 0;
    }
    const float16_t *b_head = sb;
    float16_t *c_ptr = c_head;
    uint32_t n_left = N;
    if (cputype == 55) {
      for (; n_left > 7; n_left -= 8) {
        KERNEL_M8N8_A55
        SAVE_M8N8
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    } else {
      for (; n_left > 7; n_left -= 8) {
        KERNEL_M8N8_A76
        SAVE_M8N8
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    }
    MICRO_COMPUTE_LN(8, 4, float16_t, float16_t, float16_t)
    a_head += K * 8;
    c_head += 8;
    acc_mnk += 8 * N * K;
  }

  ASSEMBLE_DUALPACK_COMPUTE_LN(4, float16_t, float16_t, float16_t, 8)
}

