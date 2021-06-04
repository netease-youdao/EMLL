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
#include "arm_neon/NeonSgemmKernel.h"
#include "arm_neon/ARMCpuType.h"
#include <sched.h>

#define NEON_SGEMM_KERNEL_M8N12_PRELOAD_A53 \
  "ldr q0,[%25]; add %25,%25,#32\n\t"\
  "ldr q3,[%26]; ldr d5,[%26,#16]; ldr x0,[%26,#24]; add %26,%26,#48\n\t"

#define NEON_SGEMM_KERNEL_M8N12_MAIN2_A53 \
  "fmov v5.d[1],x0; ldr d7,[%26,#-16]\n\t"\
  "fmla %0.4s,v0.4s,v3.s[0]; ldr x0,[%26,#-8]\n\t"\
  "fmla %2.4s,v0.4s,v3.s[1]; fmla %4.4s,v0.4s,v3.s[2]\n\t"\
  "fmov v7.d[1],x0; ldr d2,[%25,#-16]\n\t"\
  "fmla %6.4s,v0.4s,v3.s[3]; ldr x0,[%25,#-8]\n\t"\
  "fmla %8.4s,v0.4s,v5.s[0]; fmla %16.4s,v0.4s,v7.s[0]\n\t"\
  "fmov v2.d[1],x0; ldr d4,[%26]\n\t"\
  "fmla %18.4s,v0.4s,v7.s[1]; ldr x0,[%26,#8]\n\t"\
  "fmla %20.4s,v0.4s,v7.s[2]; fmla %22.4s,v0.4s,v7.s[3]\n\t"\
  "fmov v4.d[1],x0; ldr d1,[%25]\n\t"\
  "fmla %17.4s,v2.4s,v7.s[0]; ldr x0,[%25,#8]\n\t"\
  "fmla %19.4s,v2.4s,v7.s[1]; fmla %21.4s,v2.4s,v7.s[2]\n\t"\
  "fmov v1.d[1],x0; ldr d6,[%26,#16]\n\t"\
  "fmla %23.4s,v2.4s,v7.s[3]; ldr x0,[%26,#24]\n\t"\
  "fmla %9.4s,v2.4s,v5.s[0]; fmla %1.4s,v2.4s,v3.s[0]\n\t"\
  "fmov v6.d[1],x0; ldr d7,[%26,#32]\n\t"\
  "fmla %3.4s,v2.4s,v3.s[1]; ldr x0,[%26,#40]\n\t"\
  "fmla %5.4s,v2.4s,v3.s[2]; fmla %7.4s,v2.4s,v3.s[3]\n\t"\
  "fmov v7.d[1],x0; ldr d3,[%26,#48]\n\t"\
  "fmla %11.4s,v2.4s,v5.s[1]; ldr x0,[%26,#56]\n\t"\
  "fmla %13.4s,v2.4s,v5.s[2]; fmla %15.4s,v2.4s,v5.s[3]\n\t"\
  "fmov v3.d[1],x0; ldr d2,[%25,#16]\n\t"\
  "fmla %10.4s,v0.4s,v5.s[1]; ldr x0,[%25,#24]\n\t"\
  "fmla %12.4s,v0.4s,v5.s[2]; fmla %14.4s,v0.4s,v5.s[3]\n\t"\
  "fmov v2.d[1],x0; ldr d0,[%25,#32]\n\t"\
  "fmla %0.4s,v1.4s,v4.s[0]; ldr x0,[%25,#40]\n\t"\
  "fmla %2.4s,v1.4s,v4.s[1]; fmla %4.4s,v1.4s,v4.s[2]\n\t"\
  "fmov v0.d[1],x0; ldr d5,[%26,#64]\n\t"\
  "fmla %6.4s,v1.4s,v4.s[3]; ldr x0,[%26,#72]\n\t"\
  "fmla %8.4s,v1.4s,v6.s[0]; fmla %10.4s,v1.4s,v6.s[1]\n\t"\
  "add %25,%25,#64\n\t"\
  "fmla %12.4s,v1.4s,v6.s[2]\n\t"\
  "fmla %14.4s,v1.4s,v6.s[3]; fmla %16.4s,v1.4s,v7.s[0]\n\t"\
  "add %26,%26,#96\n\t"\
  "fmla %18.4s,v1.4s,v7.s[1]\n\t"\
  "fmla %20.4s,v1.4s,v7.s[2]; fmla %22.4s,v1.4s,v7.s[3]\n\t"\
  "prfm pldl1keep,[%25,#128]\n\t"\
  "fmla %1.4s,v2.4s,v4.s[0]\n\t"\
  "fmla %3.4s,v2.4s,v4.s[1]; fmla %5.4s,v2.4s,v4.s[2]\n\t"\
  "prfm pldl1keep,[%26,#192]\n\t"\
  "fmla %7.4s,v2.4s,v4.s[3]\n\t"\
  "fmla %9.4s,v2.4s,v6.s[0]; fmla %11.4s,v2.4s,v6.s[1]\n\t"\
  "sub %w24,%w24,#2\n\t"\
  "fmla %13.4s,v2.4s,v6.s[2]\n\t"\
  "fmla %15.4s,v2.4s,v6.s[3]; fmla %17.4s,v2.4s,v7.s[0]\n\t"\
  "cmp %w24,#2; prfm pldl1keep,[%26,#240]\n\t"\
  "fmla %19.4s,v2.4s,v7.s[1]\n\t"\
  "fmla %21.4s,v2.4s,v7.s[2]; fmla %23.4s,v2.4s,v7.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M8N12_TAIL2_A53 \
  "fmov v5.d[1],x0; ldr d7,[%26,#-16]\n\t"\
  "fmla %0.4s,v0.4s,v3.s[0]; ldr x0,[%26,#-8]\n\t"\
  "fmla %2.4s,v0.4s,v3.s[1]; fmla %4.4s,v0.4s,v3.s[2]\n\t"\
  "fmov v7.d[1],x0; ldr d2,[%25,#-16]\n\t"\
  "fmla %6.4s,v0.4s,v3.s[3]; ldr x0,[%25,#-8]\n\t"\
  "fmla %8.4s,v0.4s,v5.s[0]; fmla %16.4s,v0.4s,v7.s[0]\n\t"\
  "fmov v2.d[1],x0; ldr d4,[%26]\n\t"\
  "fmla %18.4s,v0.4s,v7.s[1]; ldr x0,[%26,#8]\n\t"\
  "fmla %20.4s,v0.4s,v7.s[2]; fmla %22.4s,v0.4s,v7.s[3]\n\t"\
  "fmov v4.d[1],x0; ldr d1,[%25]\n\t"\
  "fmla %17.4s,v2.4s,v7.s[0]; ldr x0,[%25,#8]\n\t"\
  "fmla %19.4s,v2.4s,v7.s[1]; fmla %21.4s,v2.4s,v7.s[2]\n\t"\
  "fmov v1.d[1],x0; ldr d6,[%26,#16]\n\t"\
  "fmla %23.4s,v2.4s,v7.s[3]; ldr x0,[%26,#24]\n\t"\
  "fmla %9.4s,v2.4s,v5.s[0]; fmla %1.4s,v2.4s,v3.s[0]\n\t"\
  "fmov v6.d[1],x0; ldr d7,[%26,#32]\n\t"\
  "fmla %3.4s,v2.4s,v3.s[1]; ldr x0,[%26,#40]\n\t"\
  "fmla %5.4s,v2.4s,v3.s[2]; fmla %7.4s,v2.4s,v3.s[3]\n\t"\
  "fmov v7.d[1],x0\n\t"\
  "fmla %11.4s,v2.4s,v5.s[1]\n\t"\
  "fmla %13.4s,v2.4s,v5.s[2]; fmla %15.4s,v2.4s,v5.s[3]\n\t"\
  "ldr d2,[%25,#16]\n\t"\
  "fmla %10.4s,v0.4s,v5.s[1]; ldr x0,[%25,#24]\n\t"\
  "fmla %12.4s,v0.4s,v5.s[2]; fmla %14.4s,v0.4s,v5.s[3]\n\t"\
  "fmov v2.d[1],x0\n\t"\
  "fmla %0.4s,v1.4s,v4.s[0]\n\t"\
  "fmla %2.4s,v1.4s,v4.s[1]; fmla %4.4s,v1.4s,v4.s[2]\n\t"\
  "fmla %6.4s,v1.4s,v4.s[3]\n\t"\
  "fmla %8.4s,v1.4s,v6.s[0]; fmla %10.4s,v1.4s,v6.s[1]\n\t"\
  "add %25,%25,#32\n\t"\
  "fmla %12.4s,v1.4s,v6.s[2]\n\t"\
  "fmla %14.4s,v1.4s,v6.s[3]; fmla %16.4s,v1.4s,v7.s[0]\n\t"\
  "add %26,%26,#48\n\t"\
  "fmla %18.4s,v1.4s,v7.s[1]\n\t"\
  "fmla %20.4s,v1.4s,v7.s[2]; fmla %22.4s,v1.4s,v7.s[3]\n\t"\
  "fmla %1.4s,v2.4s,v4.s[0]\n\t"\
  "fmla %3.4s,v2.4s,v4.s[1]; fmla %5.4s,v2.4s,v4.s[2]\n\t"\
  "fmla %7.4s,v2.4s,v4.s[3]\n\t"\
  "fmla %9.4s,v2.4s,v6.s[0]; fmla %11.4s,v2.4s,v6.s[1]\n\t"\
  "fmla %13.4s,v2.4s,v6.s[2]\n\t"\
  "fmla %15.4s,v2.4s,v6.s[3]; fmla %17.4s,v2.4s,v7.s[0]\n\t"\
  "fmla %19.4s,v2.4s,v7.s[1]\n\t"\
  "fmla %21.4s,v2.4s,v7.s[2]; fmla %23.4s,v2.4s,v7.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M8N12_TAIL1_A53 \
  "fmov v5.d[1],x0; ldr d7,[%26,#-16]\n\t"\
  "fmla %0.4s,v0.4s,v3.s[0]; ldr x0,[%26,#-8]\n\t"\
  "fmla %2.4s,v0.4s,v3.s[1]; fmla %4.4s,v0.4s,v3.s[2]\n\t"\
  "fmov v7.d[1],x0; ldr d2,[%25,#-16]\n\t"\
  "fmla %6.4s,v0.4s,v3.s[3]; ldr x0,[%25,#-8]\n\t"\
  "fmla %8.4s,v0.4s,v5.s[0]; fmla %16.4s,v0.4s,v7.s[0]\n\t"\
  "fmov v2.d[1],x0\n\t"\
  "fmla %18.4s,v0.4s,v7.s[1]\n\t"\
  "fmla %20.4s,v0.4s,v7.s[2]; fmla %22.4s,v0.4s,v7.s[3]\n\t"\
  "fmla %17.4s,v2.4s,v7.s[0]\n\t"\
  "fmla %19.4s,v2.4s,v7.s[1]; fmla %21.4s,v2.4s,v7.s[2]\n\t"\
  "fmla %23.4s,v2.4s,v7.s[3]\n\t"\
  "fmla %9.4s,v2.4s,v5.s[0]; fmla %1.4s,v2.4s,v3.s[0]\n\t"\
  "fmla %3.4s,v2.4s,v3.s[1]\n\t"\
  "fmla %5.4s,v2.4s,v3.s[2]; fmla %7.4s,v2.4s,v3.s[3]\n\t"\
  "fmla %11.4s,v2.4s,v5.s[1]\n\t"\
  "fmla %13.4s,v2.4s,v5.s[2]; fmla %15.4s,v2.4s,v5.s[3]\n\t"\
  "fmla %10.4s,v0.4s,v5.s[1]\n\t"\
  "fmla %12.4s,v0.4s,v5.s[2]; fmla %14.4s,v0.4s,v5.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M8N12_PRELOAD_A55 \
  "ldr q0,[%25]; ldr q1,[%25,#16]; add %25,%25,#32\n\t"\
  "ldr q4,[%26]; ldr d5,[%26,#16]; ldr x1,[%26,#24]; add %26,%26,#48\n\t"

#define NEON_SGEMM_KERNEL_M8N12_MAIN2_A55 \
  "fmla %0.4s,v0.4s,v4.s[0]; ldr d2,[%25]\n\t"\
  "fmla %2.4s,v0.4s,v4.s[1]; ldr x0,[%25,#8]\n\t"\
  "fmla %4.4s,v0.4s,v4.s[2]\n\t"\
  "fmla %6.4s,v0.4s,v4.s[3]; fmov v5.d[1],x1\n\t"\
  "fmla %1.4s,v1.4s,v4.s[0]; ldr d6,[%26,#-16]\n\t"\
  "fmla %3.4s,v1.4s,v4.s[1]; ldr x1,[%26,#-8]\n\t"\
  "fmla %5.4s,v1.4s,v4.s[2]\n\t"\
  "fmla %7.4s,v1.4s,v4.s[3]; fmov v2.d[1],x0\n\t"\
  "fmla %8.4s,v0.4s,v5.s[0]; ldr d3,[%25,#16]\n\t"\
  "fmla %10.4s,v0.4s,v5.s[1]; ldr x0,[%25,#24]\n\t"\
  "fmla %12.4s,v0.4s,v5.s[2]\n\t"\
  "fmla %14.4s,v0.4s,v5.s[3]; fmov v6.d[1],x1\n\t"\
  "fmla %9.4s,v1.4s,v5.s[0]; ldr d4,[%26]\n\t"\
  "fmla %11.4s,v1.4s,v5.s[1]; ldr x1,[%26,#8]\n\t"\
  "fmla %13.4s,v1.4s,v5.s[2]\n\t"\
  "fmla %15.4s,v1.4s,v5.s[3]; fmov v3.d[1],x0\n\t"\
  "fmla %16.4s,v0.4s,v6.s[0]; ldr d5,[%26,#16]\n\t"\
  "fmla %18.4s,v0.4s,v6.s[1]; ldr x0,[%26,#24]\n\t"\
  "fmla %20.4s,v0.4s,v6.s[2]\n\t"\
  "fmla %22.4s,v0.4s,v6.s[3]; fmov v4.d[1],x1\n\t"\
  "fmla %17.4s,v1.4s,v6.s[0]; add %25,%25,#64\n\t"\
  "fmla %19.4s,v1.4s,v6.s[1]; add %26,%26,#96\n\t"\
  "fmla %21.4s,v1.4s,v6.s[2]\n\t"\
  "fmla %23.4s,v1.4s,v6.s[3]\n\t"\
  "fmla %0.4s,v2.4s,v4.s[0]; ldr d0,[%25,#-32]\n\t"\
  "fmla %2.4s,v2.4s,v4.s[1]; ldr x1,[%25,#-24]\n\t"\
  "fmla %4.4s,v2.4s,v4.s[2]\n\t"\
  "fmla %6.4s,v2.4s,v4.s[3]; fmov v5.d[1],x0\n\t"\
  "fmla %1.4s,v3.4s,v4.s[0]; ldr d6,[%26,#-64]\n\t"\
  "fmla %3.4s,v3.4s,v4.s[1]; ldr x0,[%26,#-56]\n\t"\
  "fmla %5.4s,v3.4s,v4.s[2]\n\t"\
  "fmla %7.4s,v3.4s,v4.s[3]; fmov v0.d[1],x1\n\t"\
  "fmla %8.4s,v2.4s,v5.s[0]; ldr d1,[%25,#-16]\n\t"\
  "fmla %10.4s,v2.4s,v5.s[1]; ldr x1,[%25,#-8]\n\t"\
  "fmla %12.4s,v2.4s,v5.s[2]\n\t"\
  "fmla %14.4s,v2.4s,v5.s[3]; fmov v6.d[1],x0\n\t"\
  "fmla %9.4s,v3.4s,v5.s[0]; ldr d4,[%26,#-48]\n\t"\
  "fmla %11.4s,v3.4s,v5.s[1]; ldr x0,[%26,#-40]\n\t"\
  "fmla %13.4s,v3.4s,v5.s[2]\n\t"\
  "fmla %15.4s,v3.4s,v5.s[3]; fmov v1.d[1],x1\n\t"\
  "fmla %16.4s,v2.4s,v6.s[0]; ldr d5,[%26,#-32]\n\t"\
  "fmla %18.4s,v2.4s,v6.s[1]; ldr x1,[%26,#-24]\n\t"\
  "fmla %20.4s,v2.4s,v6.s[2]\n\t"\
  "fmla %22.4s,v2.4s,v6.s[3]; fmov v4.d[1],x0\n\t"\
  "fmla %17.4s,v3.4s,v6.s[0]\n\t"\
  "fmla %19.4s,v3.4s,v6.s[1]; sub %w24,%w24,#2\n\t"\
  "fmla %21.4s,v3.4s,v6.s[2]; cmp %w24,#2\n\t"\
  "fmla %23.4s,v3.4s,v6.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M8N12_TAIL2_A55 \
  "fmla %0.4s,v0.4s,v4.s[0]; ldr d2,[%25]\n\t"\
  "fmla %2.4s,v0.4s,v4.s[1]; ldr x0,[%25,#8]\n\t"\
  "fmla %4.4s,v0.4s,v4.s[2]\n\t"\
  "fmla %6.4s,v0.4s,v4.s[3]; fmov v5.d[1],x1\n\t"\
  "fmla %1.4s,v1.4s,v4.s[0]; ldr d6,[%26,#-16]\n\t"\
  "fmla %3.4s,v1.4s,v4.s[1]; ldr x1,[%26,#-8]\n\t"\
  "fmla %5.4s,v1.4s,v4.s[2]\n\t"\
  "fmla %7.4s,v1.4s,v4.s[3]; fmov v2.d[1],x0\n\t"\
  "fmla %8.4s,v0.4s,v5.s[0]; ldr d3,[%25,#16]\n\t"\
  "fmla %10.4s,v0.4s,v5.s[1]; ldr x0,[%25,#24]\n\t"\
  "fmla %12.4s,v0.4s,v5.s[2]\n\t"\
  "fmla %14.4s,v0.4s,v5.s[3]; fmov v6.d[1],x1\n\t"\
  "fmla %9.4s,v1.4s,v5.s[0]; ldr d4,[%26]\n\t"\
  "fmla %11.4s,v1.4s,v5.s[1]; ldr x1,[%26,#8]\n\t"\
  "fmla %13.4s,v1.4s,v5.s[2]\n\t"\
  "fmla %15.4s,v1.4s,v5.s[3]; fmov v3.d[1],x0\n\t"\
  "fmla %16.4s,v0.4s,v6.s[0]; ldr d5,[%26,#16]\n\t"\
  "fmla %18.4s,v0.4s,v6.s[1]; ldr x0,[%26,#24]\n\t"\
  "fmla %20.4s,v0.4s,v6.s[2]\n\t"\
  "fmla %22.4s,v0.4s,v6.s[3]; fmov v4.d[1],x1\n\t"\
  "fmla %17.4s,v1.4s,v6.s[0]; add %25,%25,#32\n\t"\
  "fmla %19.4s,v1.4s,v6.s[1]; add %26,%26,#48\n\t"\
  "fmla %21.4s,v1.4s,v6.s[2]\n\t"\
  "fmla %23.4s,v1.4s,v6.s[3]\n\t"\
  "fmla %0.4s,v2.4s,v4.s[0]\n\t"\
  "fmla %2.4s,v2.4s,v4.s[1]\n\t"\
  "fmla %4.4s,v2.4s,v4.s[2]\n\t"\
  "fmla %6.4s,v2.4s,v4.s[3]; fmov v5.d[1],x0\n\t"\
  "fmla %1.4s,v3.4s,v4.s[0]; ldr d6,[%26,#-16]\n\t"\
  "fmla %3.4s,v3.4s,v4.s[1]; ldr x0,[%26,#-8]\n\t"\
  "fmla %5.4s,v3.4s,v4.s[2]\n\t"\
  "fmla %7.4s,v3.4s,v4.s[3]\n\t"\
  "fmla %8.4s,v2.4s,v5.s[0]\n\t"\
  "fmla %10.4s,v2.4s,v5.s[1]\n\t"\
  "fmla %12.4s,v2.4s,v5.s[2]\n\t"\
  "fmla %14.4s,v2.4s,v5.s[3]; fmov v6.d[1],x0\n\t"\
  "fmla %9.4s,v3.4s,v5.s[0]\n\t"\
  "fmla %11.4s,v3.4s,v5.s[1]\n\t"\
  "fmla %13.4s,v3.4s,v5.s[2]\n\t"\
  "fmla %15.4s,v3.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v6.s[0]\n\t"\
  "fmla %18.4s,v2.4s,v6.s[1]\n\t"\
  "fmla %20.4s,v2.4s,v6.s[2]\n\t"\
  "fmla %22.4s,v2.4s,v6.s[3]\n\t"\
  "fmla %17.4s,v3.4s,v6.s[0]\n\t"\
  "fmla %19.4s,v3.4s,v6.s[1]\n\t"\
  "fmla %21.4s,v3.4s,v6.s[2]\n\t"\
  "fmla %23.4s,v3.4s,v6.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M8N12_TAIL1_A55 \
  "fmla %0.4s,v0.4s,v4.s[0]\n\t"\
  "fmla %2.4s,v0.4s,v4.s[1]\n\t"\
  "fmla %4.4s,v0.4s,v4.s[2]\n\t"\
  "fmla %6.4s,v0.4s,v4.s[3]; fmov v5.d[1],x1\n\t"\
  "fmla %1.4s,v1.4s,v4.s[0]; ldr d6,[%26,#-16]\n\t"\
  "fmla %3.4s,v1.4s,v4.s[1]; ldr x1,[%26,#-8]\n\t"\
  "fmla %5.4s,v1.4s,v4.s[2]\n\t"\
  "fmla %7.4s,v1.4s,v4.s[3]\n\t"\
  "fmla %8.4s,v0.4s,v5.s[0]\n\t"\
  "fmla %10.4s,v0.4s,v5.s[1]\n\t"\
  "fmla %12.4s,v0.4s,v5.s[2]\n\t"\
  "fmla %14.4s,v0.4s,v5.s[3]; fmov v6.d[1],x1\n\t"\
  "fmla %9.4s,v1.4s,v5.s[0]\n\t"\
  "fmla %11.4s,v1.4s,v5.s[1]\n\t"\
  "fmla %13.4s,v1.4s,v5.s[2]\n\t"\
  "fmla %15.4s,v1.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v0.4s,v6.s[0]\n\t"\
  "fmla %18.4s,v0.4s,v6.s[1]\n\t"\
  "fmla %20.4s,v0.4s,v6.s[2]\n\t"\
  "fmla %22.4s,v0.4s,v6.s[3]\n\t"\
  "fmla %17.4s,v1.4s,v6.s[0]\n\t"\
  "fmla %19.4s,v1.4s,v6.s[1]\n\t"\
  "fmla %21.4s,v1.4s,v6.s[2]\n\t"\
  "fmla %23.4s,v1.4s,v6.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M8N12_PRELOAD_A72 \
  "ldr q0,[%25]; ldr q1,[%25,#16]; add %25,%25,#32\n\t"\
  "ldr q4,[%26]; ldr q5,[%26,#16]; add %26,%26,#48\n\t"\

#define NEON_SGEMM_KERNEL_M8N12_MAIN2_A72 \
  "fmla %0.4s,v0.4s,v4.s[0]; fmla %2.4s,v0.4s,v4.s[1]; ldr q6,[%26,#-16]\n\t"\
  "fmla %4.4s,v0.4s,v4.s[2]; fmla %6.4s,v0.4s,v4.s[3]\n\t"\
  "fmla %1.4s,v1.4s,v4.s[0]; fmla %3.4s,v1.4s,v4.s[1]; ldr q2,[%25],#64\n\t"\
  "fmla %5.4s,v1.4s,v4.s[2]; fmla %7.4s,v1.4s,v4.s[3]\n\t"\
  "fmla %8.4s,v0.4s,v5.s[0]; fmla %10.4s,v0.4s,v5.s[1]; ldr q4,[%26],#96\n\t"\
  "fmla %12.4s,v0.4s,v5.s[2]; fmla %14.4s,v0.4s,v5.s[3]\n\t"\
  "fmla %9.4s,v1.4s,v5.s[0]; fmla %11.4s,v1.4s,v5.s[1]; ldr q3,[%25,#-48]\n\t"\
  "fmla %13.4s,v1.4s,v5.s[2]; fmla %15.4s,v1.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v0.4s,v6.s[0]; fmla %18.4s,v0.4s,v6.s[1]; ldr q5,[%26,#-80]\n\t"\
  "fmla %20.4s,v0.4s,v6.s[2]; fmla %22.4s,v0.4s,v6.s[3]\n\t"\
  "fmla %17.4s,v1.4s,v6.s[0]; fmla %19.4s,v1.4s,v6.s[1]; sub %w24,%w24,#2\n\t"\
  "fmla %21.4s,v1.4s,v6.s[2]; fmla %23.4s,v1.4s,v6.s[3]\n\t"\
  "fmla %0.4s,v2.4s,v4.s[0]; fmla %2.4s,v2.4s,v4.s[1]; ldr q6,[%26,#-64]\n\t"\
  "fmla %4.4s,v2.4s,v4.s[2]; fmla %6.4s,v2.4s,v4.s[3]\n\t"\
  "fmla %1.4s,v3.4s,v4.s[0]; fmla %3.4s,v3.4s,v4.s[1]; ldr q0,[%25,#-32]\n\t"\
  "fmla %5.4s,v3.4s,v4.s[2]; fmla %7.4s,v3.4s,v4.s[3]\n\t"\
  "fmla %8.4s,v2.4s,v5.s[0]; fmla %10.4s,v2.4s,v5.s[1]; ldr q4,[%26,#-48]\n\t"\
  "fmla %12.4s,v2.4s,v5.s[2]; fmla %14.4s,v2.4s,v5.s[3]\n\t"\
  "fmla %9.4s,v3.4s,v5.s[0]; fmla %11.4s,v3.4s,v5.s[1]; ldr q1,[%25,#-16]\n\t"\
  "fmla %13.4s,v3.4s,v5.s[2]; fmla %15.4s,v3.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v6.s[0]; fmla %18.4s,v2.4s,v6.s[1]; ldr q5,[%26,#-32]\n\t"\
  "fmla %20.4s,v2.4s,v6.s[2]; fmla %22.4s,v2.4s,v6.s[3]\n\t"\
  "fmla %17.4s,v3.4s,v6.s[0]; fmla %19.4s,v3.4s,v6.s[1]; cmp %w24,#2\n\t"\
  "fmla %21.4s,v3.4s,v6.s[2]; fmla %23.4s,v3.4s,v6.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M8N12_TAIL2_A72 \
  "fmla %0.4s,v0.4s,v4.s[0]; fmla %2.4s,v0.4s,v4.s[1]; ldr q6,[%26,#-16]\n\t"\
  "fmla %4.4s,v0.4s,v4.s[2]; fmla %6.4s,v0.4s,v4.s[3]\n\t"\
  "fmla %1.4s,v1.4s,v4.s[0]; fmla %3.4s,v1.4s,v4.s[1]; ldr q2,[%25],#32\n\t"\
  "fmla %5.4s,v1.4s,v4.s[2]; fmla %7.4s,v1.4s,v4.s[3]\n\t"\
  "fmla %8.4s,v0.4s,v5.s[0]; fmla %10.4s,v0.4s,v5.s[1]; ldr q4,[%26],#48\n\t"\
  "fmla %12.4s,v0.4s,v5.s[2]; fmla %14.4s,v0.4s,v5.s[3]\n\t"\
  "fmla %9.4s,v1.4s,v5.s[0]; fmla %11.4s,v1.4s,v5.s[1]; ldr q3,[%25,#-16]\n\t"\
  "fmla %13.4s,v1.4s,v5.s[2]; fmla %15.4s,v1.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v0.4s,v6.s[0]; fmla %18.4s,v0.4s,v6.s[1]; ldr q5,[%26,#-32]\n\t"\
  "fmla %20.4s,v0.4s,v6.s[2]; fmla %22.4s,v0.4s,v6.s[3]\n\t"\
  "fmla %17.4s,v1.4s,v6.s[0]; fmla %19.4s,v1.4s,v6.s[1]\n\t"\
  "fmla %21.4s,v1.4s,v6.s[2]; fmla %23.4s,v1.4s,v6.s[3]\n\t"\
  "fmla %0.4s,v2.4s,v4.s[0]; fmla %2.4s,v2.4s,v4.s[1]; ldr q6,[%26,#-16]\n\t"\
  "fmla %4.4s,v2.4s,v4.s[2]; fmla %6.4s,v2.4s,v4.s[3]\n\t"\
  "fmla %1.4s,v3.4s,v4.s[0]; fmla %3.4s,v3.4s,v4.s[1]\n\t"\
  "fmla %5.4s,v3.4s,v4.s[2]; fmla %7.4s,v3.4s,v4.s[3]\n\t"\
  "fmla %8.4s,v2.4s,v5.s[0]; fmla %10.4s,v2.4s,v5.s[1]\n\t"\
  "fmla %12.4s,v2.4s,v5.s[2]; fmla %14.4s,v2.4s,v5.s[3]\n\t"\
  "fmla %9.4s,v3.4s,v5.s[0]; fmla %11.4s,v3.4s,v5.s[1]\n\t"\
  "fmla %13.4s,v3.4s,v5.s[2]; fmla %15.4s,v3.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v6.s[0]; fmla %18.4s,v2.4s,v6.s[1]\n\t"\
  "fmla %20.4s,v2.4s,v6.s[2]; fmla %22.4s,v2.4s,v6.s[3]\n\t"\
  "fmla %17.4s,v3.4s,v6.s[0]; fmla %19.4s,v3.4s,v6.s[1]\n\t"\
  "fmla %21.4s,v3.4s,v6.s[2]; fmla %23.4s,v3.4s,v6.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M8N12_TAIL1_A72 \
  "fmla %0.4s,v0.4s,v4.s[0]; fmla %2.4s,v0.4s,v4.s[1]; ldr q6,[%26,#-16]\n\t"\
  "fmla %4.4s,v0.4s,v4.s[2]; fmla %6.4s,v0.4s,v4.s[3]\n\t"\
  "fmla %1.4s,v1.4s,v4.s[0]; fmla %3.4s,v1.4s,v4.s[1]\n\t"\
  "fmla %5.4s,v1.4s,v4.s[2]; fmla %7.4s,v1.4s,v4.s[3]\n\t"\
  "fmla %8.4s,v0.4s,v5.s[0]; fmla %10.4s,v0.4s,v5.s[1]\n\t"\
  "fmla %12.4s,v0.4s,v5.s[2]; fmla %14.4s,v0.4s,v5.s[3]\n\t"\
  "fmla %9.4s,v1.4s,v5.s[0]; fmla %11.4s,v1.4s,v5.s[1]\n\t"\
  "fmla %13.4s,v1.4s,v5.s[2]; fmla %15.4s,v1.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v0.4s,v6.s[0]; fmla %18.4s,v0.4s,v6.s[1]\n\t"\
  "fmla %20.4s,v0.4s,v6.s[2]; fmla %22.4s,v0.4s,v6.s[3]\n\t"\
  "fmla %17.4s,v1.4s,v6.s[0]; fmla %19.4s,v1.4s,v6.s[1]\n\t"\
  "fmla %21.4s,v1.4s,v6.s[2]; fmla %23.4s,v1.4s,v6.s[3]\n\t"

#define NEON_SGEMM_SAVE_M8N3_UNIT(cq1, cq2, cq3, cq4, cq5, cq6) \
  ct1 = vld1q_f32(c_tmp1); ct2 = vld1q_f32(c_tmp1 + 4);\
  ct3 = vld1q_f32(c_tmp2); ct4 = vld1q_f32(c_tmp2 + 4);\
  ct5 = vld1q_f32(c_tmp3); ct6 = vld1q_f32(c_tmp3 + 4);\
  cq1 = vfmaq_n_f32(cq1, ct1, beta); cq2 = vfmaq_n_f32(cq2, ct2, beta);\
  cq3 = vfmaq_n_f32(cq3, ct3, beta); cq4 = vfmaq_n_f32(cq4, ct4, beta);\
  cq5 = vfmaq_n_f32(cq5, ct5, beta); cq6 = vfmaq_n_f32(cq6, ct6, beta);\
  vst1q_f32(c_tmp1, cq1); vst1q_f32(c_tmp1 + 4, cq2); c_tmp1 += ldc3;\
  vst1q_f32(c_tmp2, cq3); vst1q_f32(c_tmp2 + 4, cq4); c_tmp2 += ldc3;\
  vst1q_f32(c_tmp3, cq5); vst1q_f32(c_tmp3 + 4, cq6); c_tmp3 += ldc3;

#define NEON_SGEMM_SAVE_M8N12_ASM1 \
  float *c_tmp1 = c_ptr;\
  float *c_tmp2 = c_ptr + ldc;\
  float *c_tmp3 = c_ptr + ldc * 2;\
  uint32_t ldc3 = ldc * 3;\
  float32x4_t ct1, ct2, ct3, ct4, ct5, ct6;\
  NEON_SGEMM_SAVE_M8N3_UNIT(cq01, cq02, cq03, cq04, cq05, cq06)\
  NEON_SGEMM_SAVE_M8N3_UNIT(cq07, cq08, cq09, cq10, cq11, cq12)\
  NEON_SGEMM_SAVE_M8N3_UNIT(cq13, cq14, cq15, cq16, cq17, cq18)\
  NEON_SGEMM_SAVE_M8N3_UNIT(cq19, cq20, cq21, cq22, cq23, cq24)

#define NEON_SGEMM_KERNEL_M12N8_PRELOAD_A53 \
  "ldr q5,[%26]; add %26,%26,#32\n\t"\
  "ldr q0,[%25]; ldr d2,[%25,#16]; ldr x0,[%25,#24]; add %25,%25,#48\n\t"

#define NEON_SGEMM_KERNEL_M12N8_MAIN2_A53 \
  "fmov v2.d[1],x0; ldr d4,[%25,#-16]\n\t"\
  "fmla %0.4s,v0.4s,v5.s[0]; ldr x0,[%25,#-8]\n\t"\
  "fmla %1.4s,v0.4s,v5.s[1]; fmla %2.4s,v0.4s,v5.s[2]\n\t"\
  "fmov v4.d[1],x0; ldr d7,[%26,#-16]\n\t"\
  "fmla %3.4s,v0.4s,v5.s[3]; ldr x0,[%26,#-8]\n\t"\
  "fmla %8.4s,v2.4s,v5.s[0]; fmla %16.4s,v4.4s,v5.s[0]\n\t"\
  "fmov v7.d[1],x0; ldr d6,[%26]\n\t"\
  "fmla %17.4s,v4.4s,v5.s[1]; ldr x0,[%26,#8]\n\t"\
  "fmla %18.4s,v4.4s,v5.s[2]; fmla %20.4s,v4.4s,v7.s[0]\n\t"\
  "fmov v6.d[1],x0; ldr d1,[%25]\n\t"\
  "fmla %21.4s,v4.4s,v7.s[1]; ldr x0,[%25,#8]\n\t"\
  "fmla %22.4s,v4.4s,v7.s[2]; fmla %23.4s,v4.4s,v7.s[3]\n\t"\
  "fmov v1.d[1],x0; ldr d3,[%25,#16]\n\t"\
  "fmla %19.4s,v4.4s,v5.s[3]; ldr x0,[%25,#24]\n\t"\
  "fmla %4.4s,v0.4s,v7.s[0]; fmla %5.4s,v0.4s,v7.s[1]\n\t"\
  "fmov v3.d[1],x0; ldr d4,[%25,#32]\n\t"\
  "fmla %6.4s,v0.4s,v7.s[2]; ldr x0,[%25,#40]\n\t"\
  "fmla %7.4s,v0.4s,v7.s[3]; fmla %12.4s,v2.4s,v7.s[0]\n\t"\
  "fmov v4.d[1],x0; ldr d0,[%25,#48]\n\t"\
  "fmla %13.4s,v2.4s,v7.s[1]; ldr x0,[%25,#56]\n\t"\
  "fmla %14.4s,v2.4s,v7.s[2]; fmla %15.4s,v2.4s,v7.s[3]\n\t"\
  "fmov v0.d[1],x0; ldr d7,[%26,#16]\n\t"\
  "fmla %9.4s,v2.4s,v5.s[1]; ldr x0,[%26,#24]\n\t"\
  "fmla %10.4s,v2.4s,v5.s[2]; fmla %11.4s,v2.4s,v5.s[3]\n\t"\
  "fmov v7.d[1],x0; ldr d5,[%26,#32]\n\t"\
  "fmla %0.4s,v1.4s,v6.s[0]; ldr x0,[%26,#40]\n\t"\
  "fmla %1.4s,v1.4s,v6.s[1]; fmla %2.4s,v1.4s,v6.s[2]\n\t"\
  "fmov v5.d[1],x0; ldr d2,[%25,#64]\n\t"\
  "fmla %3.4s,v1.4s,v6.s[3]; ldr x0,[%25,#72]\n\t"\
  "fmla %4.4s,v1.4s,v7.s[0]; fmla %5.4s,v1.4s,v7.s[1]\n\t"\
  "add %25,%25,#96\n\t"\
  "fmla %6.4s,v1.4s,v7.s[2]\n\t"\
  "fmla %7.4s,v1.4s,v7.s[3]; fmla %8.4s,v3.4s,v6.s[0]\n\t"\
  "prfm pldl1keep,[%25,#192]\n\t"\
  "fmla %9.4s,v3.4s,v6.s[1]\n\t"\
  "fmla %10.4s,v3.4s,v6.s[2]; fmla %11.4s,v3.4s,v6.s[3]\n\t"\
  "add %26,%26,#64\n\t"\
  "fmla %12.4s,v3.4s,v7.s[0]\n\t"\
  "fmla %13.4s,v3.4s,v7.s[1]; fmla %14.4s,v3.4s,v7.s[2]\n\t"\
  "prfm pldl1keep,[%26,#128]\n\t"\
  "fmla %15.4s,v3.4s,v7.s[3]\n\t"\
  "fmla %16.4s,v4.4s,v6.s[0]; fmla %17.4s,v4.4s,v6.s[1]\n\t"\
  "sub %w24,%w24,#2\n\t"\
  "fmla %18.4s,v4.4s,v6.s[2]\n\t"\
  "fmla %19.4s,v4.4s,v6.s[3]; fmla %20.4s,v4.4s,v7.s[0]\n\t"\
  "cmp %w24,#2; prfm pldl1keep,[%25,#240]\n\t"\
  "fmla %21.4s,v4.4s,v7.s[1]\n\t"\
  "fmla %22.4s,v4.4s,v7.s[2]; fmla %23.4s,v4.4s,v7.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M12N8_TAIL2_A53 \
  "fmov v2.d[1],x0; ldr d4,[%25,#-16]\n\t"\
  "fmla %0.4s,v0.4s,v5.s[0]; ldr x0,[%25,#-8]\n\t"\
  "fmla %1.4s,v0.4s,v5.s[1]; fmla %2.4s,v0.4s,v5.s[2]\n\t"\
  "fmov v4.d[1],x0; ldr d7,[%26,#-16]\n\t"\
  "fmla %3.4s,v0.4s,v5.s[3]; ldr x0,[%26,#-8]\n\t"\
  "fmla %8.4s,v2.4s,v5.s[0]; fmla %16.4s,v4.4s,v5.s[0]\n\t"\
  "fmov v7.d[1],x0; ldr d6,[%26]\n\t"\
  "fmla %17.4s,v4.4s,v5.s[1]; ldr x0,[%26,#8]\n\t"\
  "fmla %18.4s,v4.4s,v5.s[2]; fmla %20.4s,v4.4s,v7.s[0]\n\t"\
  "fmov v6.d[1],x0; ldr d1,[%25]\n\t"\
  "fmla %21.4s,v4.4s,v7.s[1]; ldr x0,[%25,#8]\n\t"\
  "fmla %22.4s,v4.4s,v7.s[2]; fmla %23.4s,v4.4s,v7.s[3]\n\t"\
  "fmov v1.d[1],x0; ldr d3,[%25,#16]\n\t"\
  "fmla %19.4s,v4.4s,v5.s[3]; ldr x0,[%25,#24]\n\t"\
  "fmla %4.4s,v0.4s,v7.s[0]; fmla %5.4s,v0.4s,v7.s[1]\n\t"\
  "fmov v3.d[1],x0; ldr d4,[%25,#32]\n\t"\
  "fmla %6.4s,v0.4s,v7.s[2]; ldr x0,[%25,#40]\n\t"\
  "fmla %7.4s,v0.4s,v7.s[3]; fmla %12.4s,v2.4s,v7.s[0]\n\t"\
  "fmov v4.d[1],x0\n\t"\
  "fmla %13.4s,v2.4s,v7.s[1]\n\t"\
  "fmla %14.4s,v2.4s,v7.s[2]; fmla %15.4s,v2.4s,v7.s[3]\n\t"\
  "ldr d7,[%26,#16]\n\t"\
  "fmla %9.4s,v2.4s,v5.s[1]; ldr x0,[%26,#24]\n\t"\
  "fmla %10.4s,v2.4s,v5.s[2]; fmla %11.4s,v2.4s,v5.s[3]\n\t"\
  "fmov v7.d[1],x0\n\t"\
  "fmla %0.4s,v1.4s,v6.s[0]\n\t"\
  "fmla %1.4s,v1.4s,v6.s[1]; fmla %2.4s,v1.4s,v6.s[2]\n\t"\
  "fmla %3.4s,v1.4s,v6.s[3]\n\t"\
  "fmla %4.4s,v1.4s,v7.s[0]; fmla %5.4s,v1.4s,v7.s[1]\n\t"\
  "add %25,%25,#48\n\t"\
  "fmla %6.4s,v1.4s,v7.s[2]\n\t"\
  "fmla %7.4s,v1.4s,v7.s[3]; fmla %8.4s,v3.4s,v6.s[0]\n\t"\
  "add %26,%26,#32\n\t"\
  "fmla %9.4s,v3.4s,v6.s[1]\n\t"\
  "fmla %10.4s,v3.4s,v6.s[2]; fmla %11.4s,v3.4s,v6.s[3]\n\t"\
  "fmla %12.4s,v3.4s,v7.s[0]\n\t"\
  "fmla %13.4s,v3.4s,v7.s[1]; fmla %14.4s,v3.4s,v7.s[2]\n\t"\
  "fmla %15.4s,v3.4s,v7.s[3]\n\t"\
  "fmla %16.4s,v4.4s,v6.s[0]; fmla %17.4s,v4.4s,v6.s[1]\n\t"\
  "fmla %18.4s,v4.4s,v6.s[2]\n\t"\
  "fmla %19.4s,v4.4s,v6.s[3]; fmla %20.4s,v4.4s,v7.s[0]\n\t"\
  "fmla %21.4s,v4.4s,v7.s[1]\n\t"\
  "fmla %22.4s,v4.4s,v7.s[2]; fmla %23.4s,v4.4s,v7.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M12N8_TAIL1_A53 \
  "fmov v2.d[1],x0; ldr d4,[%25,#-16]\n\t"\
  "fmla %0.4s,v0.4s,v5.s[0]; ldr x0,[%25,#-8]\n\t"\
  "fmla %1.4s,v0.4s,v5.s[1]; fmla %2.4s,v0.4s,v5.s[2]\n\t"\
  "fmov v4.d[1],x0; ldr d7,[%26,#-16]\n\t"\
  "fmla %3.4s,v0.4s,v5.s[3]; ldr x0,[%26,#-8]\n\t"\
  "fmla %8.4s,v2.4s,v5.s[0]; fmla %16.4s,v4.4s,v5.s[0]\n\t"\
  "fmov v7.d[1],x0\n\t"\
  "fmla %17.4s,v4.4s,v5.s[1]\n\t"\
  "fmla %18.4s,v4.4s,v5.s[2]; fmla %20.4s,v4.4s,v7.s[0]\n\t"\
  "fmla %21.4s,v4.4s,v7.s[1]\n\t"\
  "fmla %22.4s,v4.4s,v7.s[2]; fmla %23.4s,v4.4s,v7.s[3]\n\t"\
  "fmla %19.4s,v4.4s,v5.s[3]\n\t"\
  "fmla %4.4s,v0.4s,v7.s[0]; fmla %5.4s,v0.4s,v7.s[1]\n\t"\
  "fmla %6.4s,v0.4s,v7.s[2]\n\t"\
  "fmla %7.4s,v0.4s,v7.s[3]; fmla %12.4s,v2.4s,v7.s[0]\n\t"\
  "fmla %13.4s,v2.4s,v7.s[1]\n\t"\
  "fmla %14.4s,v2.4s,v7.s[2]; fmla %15.4s,v2.4s,v7.s[3]\n\t"\
  "fmla %9.4s,v2.4s,v5.s[1]\n\t"\
  "fmla %10.4s,v2.4s,v5.s[2]; fmla %11.4s,v2.4s,v5.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M12N8_PRELOAD_A55 \
  "ldr q4,[%26]; ldr q5,[%26,#16]; add %26,%26,#32\n\t"\
  "ldr q0,[%25]; ldr d1,[%25,#16]; ldr x1,[%25,#24]; add %25,%25,#48\n\t"

#define NEON_SGEMM_KERNEL_M12N8_MAIN2_A55 \
  "fmla %0.4s,v0.4s,v4.s[0]; ldr d6,[%26]\n\t"\
  "fmla %1.4s,v0.4s,v4.s[1]; ldr x0,[%26,#8]\n\t"\
  "fmla %2.4s,v0.4s,v4.s[2]\n\t"\
  "fmla %3.4s,v0.4s,v4.s[3]; fmov v1.d[1],x1\n\t"\
  "fmla %4.4s,v0.4s,v5.s[0]; ldr d2,[%25,#-16]\n\t"\
  "fmla %5.4s,v0.4s,v5.s[1]; ldr x1,[%25,#-8]\n\t"\
  "fmla %6.4s,v0.4s,v5.s[2]\n\t"\
  "fmla %7.4s,v0.4s,v5.s[3]; fmov v6.d[1],x0\n\t"\
  "fmla %8.4s,v1.4s,v4.s[0]; ldr d7,[%26,#16]\n\t"\
  "fmla %9.4s,v1.4s,v4.s[1]; ldr x0,[%26,#24]\n\t"\
  "fmla %10.4s,v1.4s,v4.s[2]\n\t"\
  "fmla %11.4s,v1.4s,v4.s[3]; fmov v2.d[1],x1\n\t"\
  "fmla %12.4s,v1.4s,v5.s[0]; ldr d0,[%25]\n\t"\
  "fmla %13.4s,v1.4s,v5.s[1]; ldr x1,[%25,#8]\n\t"\
  "fmla %14.4s,v1.4s,v5.s[2]\n\t"\
  "fmla %15.4s,v1.4s,v5.s[3]; fmov v7.d[1],x0\n\t"\
  "fmla %16.4s,v2.4s,v4.s[0]; ldr d1,[%25,#16]\n\t"\
  "fmla %17.4s,v2.4s,v4.s[1]; ldr x0,[%25,#24]\n\t"\
  "fmla %18.4s,v2.4s,v4.s[2]\n\t"\
  "fmla %19.4s,v2.4s,v4.s[3]; fmov v0.d[1],x1\n\t"\
  "fmla %20.4s,v2.4s,v5.s[0]; add %25,%25,#96\n\t"\
  "fmla %21.4s,v2.4s,v5.s[1]; add %26,%26,#64\n\t"\
  "fmla %22.4s,v2.4s,v5.s[2]\n\t"\
  "fmla %23.4s,v2.4s,v5.s[3]\n\t"\
  "fmla %0.4s,v0.4s,v6.s[0]; ldr d4,[%26,#-32]\n\t"\
  "fmla %1.4s,v0.4s,v6.s[1]; ldr x1,[%26,#-24]\n\t"\
  "fmla %2.4s,v0.4s,v6.s[2]\n\t"\
  "fmla %3.4s,v0.4s,v6.s[3]; fmov v1.d[1],x0\n\t"\
  "fmla %4.4s,v0.4s,v7.s[0]; ldr d2,[%25,#-64]\n\t"\
  "fmla %5.4s,v0.4s,v7.s[1]; ldr x0,[%25,#-56]\n\t"\
  "fmla %6.4s,v0.4s,v7.s[2]\n\t"\
  "fmla %7.4s,v0.4s,v7.s[3]; fmov v4.d[1],x1\n\t"\
  "fmla %8.4s,v1.4s,v6.s[0]; ldr d5,[%26,#-16]\n\t"\
  "fmla %9.4s,v1.4s,v6.s[1]; ldr x1,[%26,#-8]\n\t"\
  "fmla %10.4s,v1.4s,v6.s[2]\n\t"\
  "fmla %11.4s,v1.4s,v6.s[3]; fmov v2.d[1],x0\n\t"\
  "fmla %12.4s,v1.4s,v7.s[0]; ldr d0,[%25,#-48]\n\t"\
  "fmla %13.4s,v1.4s,v7.s[1]; ldr x0,[%25,#-40]\n\t"\
  "fmla %14.4s,v1.4s,v7.s[2]\n\t"\
  "fmla %15.4s,v1.4s,v7.s[3]; fmov v5.d[1],x1\n\t"\
  "fmla %16.4s,v2.4s,v6.s[0]; ldr d1,[%25,#-32]\n\t"\
  "fmla %17.4s,v2.4s,v6.s[1]; ldr x1,[%25,#-24]\n\t"\
  "fmla %18.4s,v2.4s,v6.s[2]\n\t"\
  "fmla %19.4s,v2.4s,v6.s[3]; fmov v0.d[1],x0\n\t"\
  "fmla %20.4s,v2.4s,v7.s[0]\n\t"\
  "fmla %21.4s,v2.4s,v7.s[1]; sub %w24,%w24,#2\n\t"\
  "fmla %22.4s,v2.4s,v7.s[2]; cmp %w24,#2\n\t"\
  "fmla %23.4s,v2.4s,v7.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M12N8_TAIL2_A55 \
  "fmla %0.4s,v0.4s,v4.s[0]; ldr d6,[%26]\n\t"\
  "fmla %1.4s,v0.4s,v4.s[1]; ldr x0,[%26,#8]\n\t"\
  "fmla %2.4s,v0.4s,v4.s[2]\n\t"\
  "fmla %3.4s,v0.4s,v4.s[3]; fmov v1.d[1],x1\n\t"\
  "fmla %4.4s,v0.4s,v5.s[0]; ldr d2,[%25,#-16]\n\t"\
  "fmla %5.4s,v0.4s,v5.s[1]; ldr x1,[%25,#-8]\n\t"\
  "fmla %6.4s,v0.4s,v5.s[2]\n\t"\
  "fmla %7.4s,v0.4s,v5.s[3]; fmov v6.d[1],x0\n\t"\
  "fmla %8.4s,v1.4s,v4.s[0]; ldr d7,[%26,#16]\n\t"\
  "fmla %9.4s,v1.4s,v4.s[1]; ldr x0,[%26,#24]\n\t"\
  "fmla %10.4s,v1.4s,v4.s[2]\n\t"\
  "fmla %11.4s,v1.4s,v4.s[3]; fmov v2.d[1],x1\n\t"\
  "fmla %12.4s,v1.4s,v5.s[0]; ldr d0,[%25]\n\t"\
  "fmla %13.4s,v1.4s,v5.s[1]; ldr x1,[%25,#8]\n\t"\
  "fmla %14.4s,v1.4s,v5.s[2]\n\t"\
  "fmla %15.4s,v1.4s,v5.s[3]; fmov v7.d[1],x0\n\t"\
  "fmla %16.4s,v2.4s,v4.s[0]; ldr d1,[%25,#16]\n\t"\
  "fmla %17.4s,v2.4s,v4.s[1]; ldr x0,[%25,#24]\n\t"\
  "fmla %18.4s,v2.4s,v4.s[2]\n\t"\
  "fmla %19.4s,v2.4s,v4.s[3]; fmov v0.d[1],x1\n\t"\
  "fmla %20.4s,v2.4s,v5.s[0]; add %25,%25,#48\n\t"\
  "fmla %21.4s,v2.4s,v5.s[1]; add %26,%26,#32\n\t"\
  "fmla %22.4s,v2.4s,v5.s[2]\n\t"\
  "fmla %23.4s,v2.4s,v5.s[3]\n\t"\
  "fmla %0.4s,v0.4s,v6.s[0]\n\t"\
  "fmla %1.4s,v0.4s,v6.s[1]\n\t"\
  "fmla %2.4s,v0.4s,v6.s[2]\n\t"\
  "fmla %3.4s,v0.4s,v6.s[3]; fmov v1.d[1],x0\n\t"\
  "fmla %4.4s,v0.4s,v7.s[0]; ldr d2,[%25,#-16]\n\t"\
  "fmla %5.4s,v0.4s,v7.s[1]; ldr x0,[%25,#-8]\n\t"\
  "fmla %6.4s,v0.4s,v7.s[2]\n\t"\
  "fmla %7.4s,v0.4s,v7.s[3]\n\t"\
  "fmla %8.4s,v1.4s,v6.s[0]\n\t"\
  "fmla %9.4s,v1.4s,v6.s[1]\n\t"\
  "fmla %10.4s,v1.4s,v6.s[2]\n\t"\
  "fmla %11.4s,v1.4s,v6.s[3]; fmov v2.d[1],x0\n\t"\
  "fmla %12.4s,v1.4s,v7.s[0]\n\t"\
  "fmla %13.4s,v1.4s,v7.s[1]\n\t"\
  "fmla %14.4s,v1.4s,v7.s[2]\n\t"\
  "fmla %15.4s,v1.4s,v7.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v6.s[0]\n\t"\
  "fmla %17.4s,v2.4s,v6.s[1]\n\t"\
  "fmla %18.4s,v2.4s,v6.s[2]\n\t"\
  "fmla %19.4s,v2.4s,v6.s[3]\n\t"\
  "fmla %20.4s,v2.4s,v7.s[0]\n\t"\
  "fmla %21.4s,v2.4s,v7.s[1]\n\t"\
  "fmla %22.4s,v2.4s,v7.s[2]\n\t"\
  "fmla %23.4s,v2.4s,v7.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M12N8_TAIL1_A55 \
  "fmla %0.4s,v0.4s,v4.s[0]\n\t"\
  "fmla %1.4s,v0.4s,v4.s[1]\n\t"\
  "fmla %2.4s,v0.4s,v4.s[2]\n\t"\
  "fmla %3.4s,v0.4s,v4.s[3]; fmov v1.d[1],x1\n\t"\
  "fmla %4.4s,v0.4s,v5.s[0]; ldr d2,[%25,#-16]\n\t"\
  "fmla %5.4s,v0.4s,v5.s[1]; ldr x1,[%25,#-8]\n\t"\
  "fmla %6.4s,v0.4s,v5.s[2]\n\t"\
  "fmla %7.4s,v0.4s,v5.s[3]\n\t"\
  "fmla %8.4s,v1.4s,v4.s[0]\n\t"\
  "fmla %9.4s,v1.4s,v4.s[1]\n\t"\
  "fmla %10.4s,v1.4s,v4.s[2]\n\t"\
  "fmla %11.4s,v1.4s,v4.s[3]; fmov v2.d[1],x1\n\t"\
  "fmla %12.4s,v1.4s,v5.s[0]\n\t"\
  "fmla %13.4s,v1.4s,v5.s[1]\n\t"\
  "fmla %14.4s,v1.4s,v5.s[2]\n\t"\
  "fmla %15.4s,v1.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v4.s[0]\n\t"\
  "fmla %17.4s,v2.4s,v4.s[1]\n\t"\
  "fmla %18.4s,v2.4s,v4.s[2]\n\t"\
  "fmla %19.4s,v2.4s,v4.s[3]\n\t"\
  "fmla %20.4s,v2.4s,v5.s[0]\n\t"\
  "fmla %21.4s,v2.4s,v5.s[1]\n\t"\
  "fmla %22.4s,v2.4s,v5.s[2]\n\t"\
  "fmla %23.4s,v2.4s,v5.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M12N8_PRELOAD_A72 \
  "ldr q0,[%25]; ldr q1,[%25,#16]; add %25,%25,#48\n\t"\
  "ldr q4,[%26]; ldr q5,[%26,#16]; add %26,%26,#32\n\t"

#define NEON_SGEMM_KERNEL_M12N8_MAIN2_A72 \
  "fmla %0.4s,v0.4s,v4.s[0]; fmla %1.4s,v0.4s,v4.s[1]; ldr q2,[%25,#-16]\n\t"\
  "fmla %2.4s,v0.4s,v4.s[2]; fmla %3.4s,v0.4s,v4.s[3]\n\t"\
  "fmla %4.4s,v0.4s,v5.s[0]; fmla %5.4s,v0.4s,v5.s[1]; ldr q6,[%26],#64\n\t"\
  "fmla %6.4s,v0.4s,v5.s[2]; fmla %7.4s,v0.4s,v5.s[3]\n\t"\
  "fmla %8.4s,v1.4s,v4.s[0]; fmla %9.4s,v1.4s,v4.s[1]; ldr q0,[%25],#96\n\t"\
  "fmla %10.4s,v1.4s,v4.s[2]; fmla %11.4s,v1.4s,v4.s[3]\n\t"\
  "fmla %12.4s,v1.4s,v5.s[0]; fmla %13.4s,v1.4s,v5.s[1]; ldr q7,[%26,#-48]\n\t"\
  "fmla %14.4s,v1.4s,v5.s[2]; fmla %15.4s,v1.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v4.s[0]; fmla %17.4s,v2.4s,v4.s[1]; ldr q1,[%25,#-80]\n\t"\
  "fmla %18.4s,v2.4s,v4.s[2]; fmla %19.4s,v2.4s,v4.s[3]\n\t"\
  "fmla %20.4s,v2.4s,v5.s[0]; fmla %21.4s,v2.4s,v5.s[1]; sub %w24,%w24,#2\n\t"\
  "fmla %22.4s,v2.4s,v5.s[2]; fmla %23.4s,v2.4s,v5.s[3]\n\t"\
  "fmla %0.4s,v0.4s,v6.s[0]; fmla %1.4s,v0.4s,v6.s[1]; ldr q2,[%25,#-64]\n\t"\
  "fmla %2.4s,v0.4s,v6.s[2]; fmla %3.4s,v0.4s,v6.s[3]\n\t"\
  "fmla %4.4s,v0.4s,v7.s[0]; fmla %5.4s,v0.4s,v7.s[1]; ldr q4,[%26,#-32]\n\t"\
  "fmla %6.4s,v0.4s,v7.s[2]; fmla %7.4s,v0.4s,v7.s[3]\n\t"\
  "fmla %8.4s,v1.4s,v6.s[0]; fmla %9.4s,v1.4s,v6.s[1]; ldr q0,[%25,#-48]\n\t"\
  "fmla %10.4s,v1.4s,v6.s[2]; fmla %11.4s,v1.4s,v6.s[3]\n\t"\
  "fmla %12.4s,v1.4s,v7.s[0]; fmla %13.4s,v1.4s,v7.s[1]; ldr q5,[%26,#-16]\n\t"\
  "fmla %14.4s,v1.4s,v7.s[2]; fmla %15.4s,v1.4s,v7.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v6.s[0]; fmla %17.4s,v2.4s,v6.s[1]; ldr q1,[%25,#-32]\n\t"\
  "fmla %18.4s,v2.4s,v6.s[2]; fmla %19.4s,v2.4s,v6.s[3]\n\t"\
  "fmla %20.4s,v2.4s,v7.s[0]; fmla %21.4s,v2.4s,v7.s[1]; cmp %w24,#2\n\t"\
  "fmla %22.4s,v2.4s,v7.s[2]; fmla %23.4s,v2.4s,v7.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M12N8_TAIL2_A72 \
  "fmla %0.4s,v0.4s,v4.s[0]; fmla %1.4s,v0.4s,v4.s[1]; ldr q2,[%25,#-16]\n\t"\
  "fmla %2.4s,v0.4s,v4.s[2]; fmla %3.4s,v0.4s,v4.s[3]\n\t"\
  "fmla %4.4s,v0.4s,v5.s[0]; fmla %5.4s,v0.4s,v5.s[1]; ldr q6,[%26],#32\n\t"\
  "fmla %6.4s,v0.4s,v5.s[2]; fmla %7.4s,v0.4s,v5.s[3]\n\t"\
  "fmla %8.4s,v1.4s,v4.s[0]; fmla %9.4s,v1.4s,v4.s[1]; ldr q0,[%25],#48\n\t"\
  "fmla %10.4s,v1.4s,v4.s[2]; fmla %11.4s,v1.4s,v4.s[3]\n\t"\
  "fmla %12.4s,v1.4s,v5.s[0]; fmla %13.4s,v1.4s,v5.s[1]; ldr q7,[%26,#-16]\n\t"\
  "fmla %14.4s,v1.4s,v5.s[2]; fmla %15.4s,v1.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v4.s[0]; fmla %17.4s,v2.4s,v4.s[1]; ldr q1,[%25,#-32]\n\t"\
  "fmla %18.4s,v2.4s,v4.s[2]; fmla %19.4s,v2.4s,v4.s[3]\n\t"\
  "fmla %20.4s,v2.4s,v5.s[0]; fmla %21.4s,v2.4s,v5.s[1]\n\t"\
  "fmla %22.4s,v2.4s,v5.s[2]; fmla %23.4s,v2.4s,v5.s[3]\n\t"\
  "fmla %0.4s,v0.4s,v6.s[0]; fmla %1.4s,v0.4s,v6.s[1]; ldr q2,[%25,#-16]\n\t"\
  "fmla %2.4s,v0.4s,v6.s[2]; fmla %3.4s,v0.4s,v6.s[3]\n\t"\
  "fmla %4.4s,v0.4s,v7.s[0]; fmla %5.4s,v0.4s,v7.s[1]\n\t"\
  "fmla %6.4s,v0.4s,v7.s[2]; fmla %7.4s,v0.4s,v7.s[3]\n\t"\
  "fmla %8.4s,v1.4s,v6.s[0]; fmla %9.4s,v1.4s,v6.s[1]\n\t"\
  "fmla %10.4s,v1.4s,v6.s[2]; fmla %11.4s,v1.4s,v6.s[3]\n\t"\
  "fmla %12.4s,v1.4s,v7.s[0]; fmla %13.4s,v1.4s,v7.s[1]\n\t"\
  "fmla %14.4s,v1.4s,v7.s[2]; fmla %15.4s,v1.4s,v7.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v6.s[0]; fmla %17.4s,v2.4s,v6.s[1]\n\t"\
  "fmla %18.4s,v2.4s,v6.s[2]; fmla %19.4s,v2.4s,v6.s[3]\n\t"\
  "fmla %20.4s,v2.4s,v7.s[0]; fmla %21.4s,v2.4s,v7.s[1]\n\t"\
  "fmla %22.4s,v2.4s,v7.s[2]; fmla %23.4s,v2.4s,v7.s[3]\n\t"

#define NEON_SGEMM_KERNEL_M12N8_TAIL1_A72 \
  "fmla %0.4s,v0.4s,v4.s[0]; fmla %1.4s,v0.4s,v4.s[1]; ldr q2,[%25,#-16]\n\t"\
  "fmla %2.4s,v0.4s,v4.s[2]; fmla %3.4s,v0.4s,v4.s[3]\n\t"\
  "fmla %4.4s,v0.4s,v5.s[0]; fmla %5.4s,v0.4s,v5.s[1]\n\t"\
  "fmla %6.4s,v0.4s,v5.s[2]; fmla %7.4s,v0.4s,v5.s[3]\n\t"\
  "fmla %8.4s,v1.4s,v4.s[0]; fmla %9.4s,v1.4s,v4.s[1]\n\t"\
  "fmla %10.4s,v1.4s,v4.s[2]; fmla %11.4s,v1.4s,v4.s[3]\n\t"\
  "fmla %12.4s,v1.4s,v5.s[0]; fmla %13.4s,v1.4s,v5.s[1]\n\t"\
  "fmla %14.4s,v1.4s,v5.s[2]; fmla %15.4s,v1.4s,v5.s[3]\n\t"\
  "fmla %16.4s,v2.4s,v4.s[0]; fmla %17.4s,v2.4s,v4.s[1]\n\t"\
  "fmla %18.4s,v2.4s,v4.s[2]; fmla %19.4s,v2.4s,v4.s[3]\n\t"\
  "fmla %20.4s,v2.4s,v5.s[0]; fmla %21.4s,v2.4s,v5.s[1]\n\t"\
  "fmla %22.4s,v2.4s,v5.s[2]; fmla %23.4s,v2.4s,v5.s[3]\n\t"

#define NEON_SGEMM_SAVE_M12N2_UNIT(cq1, cq2, cq3, cq4, cq5, cq6) \
  ct1 = vld1q_f32(c_tmp1);\
  ct2 = vld1q_f32(c_tmp1 + 4);\
  ct3 = vld1q_f32(c_tmp1 + 8);\
  ct4 = vld1q_f32(c_tmp2);\
  ct5 = vld1q_f32(c_tmp2 + 4);\
  ct6 = vld1q_f32(c_tmp2 + 8);\
  cq1 = vfmaq_n_f32(cq1, ct1, beta); cq2 = vfmaq_n_f32(cq2, ct2, beta);\
  cq3 = vfmaq_n_f32(cq3, ct3, beta); cq4 = vfmaq_n_f32(cq4, ct4, beta);\
  cq5 = vfmaq_n_f32(cq5, ct5, beta); cq6 = vfmaq_n_f32(cq6, ct6, beta);\
  vst1q_f32(c_tmp1, cq1);\
  vst1q_f32(c_tmp1 + 4, cq2);\
  vst1q_f32(c_tmp1 + 8, cq3); c_tmp1 += ldc2;\
  vst1q_f32(c_tmp2, cq4);\
  vst1q_f32(c_tmp2 + 4, cq5);\
  vst1q_f32(c_tmp2 + 8, cq6); c_tmp2 += ldc2;

#define NEON_SGEMM_SAVE_M12N8_ASM1 \
  float *c_tmp1 = c_ptr;\
  float *c_tmp2 = c_ptr + ldc;\
  uint32_t ldc2 = ldc * 2;\
  float32x4_t ct1, ct2, ct3, ct4, ct5, ct6;\
  NEON_SGEMM_SAVE_M12N2_UNIT(cq01, cq09, cq17, cq02, cq10, cq18)\
  NEON_SGEMM_SAVE_M12N2_UNIT(cq03, cq11, cq19, cq04, cq12, cq20)\
  NEON_SGEMM_SAVE_M12N2_UNIT(cq05, cq13, cq21, cq06, cq14, cq22)\
  NEON_SGEMM_SAVE_M12N2_UNIT(cq07, cq15, cq23, cq08, cq16, cq24)

#define PREF_C_1_LANE(n, mdim) \
  pref_c(c_pref); pref_c(c_pref + mdim - 1); c_pref += ldc;
#define PREF_C(mdim, ndim) \
  MACRO_EXPANSION_##ndim(VOID_BASE, PREF_C_1_LANE, mdim)

#define NEON_SGEMM_COMPUTE_ASM1(mdim, ndim, cputype) \
  float *c_pref = c_ptr; PREF_C(mdim, ndim)\
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  uint32_t k_left = K;\
  float32x4_t cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  float32x4_t cq09, cq10, cq11, cq12, cq13, cq14, cq15, cq16;\
  float32x4_t cq17, cq18, cq19, cq20, cq21, cq22, cq23, cq24;\
  __asm__ __volatile__ (\
    "movi %0.16b,#0; movi %1.16b,#0\n\t"\
    "mov %2.16b,%0.16b; mov %3.16b,%1.16b\n\t"\
    "mov %4.16b,%0.16b; mov %5.16b,%1.16b\n\t"\
    "mov %6.16b,%0.16b; mov %7.16b,%1.16b\n\t"\
    "mov %8.16b,%0.16b; mov %9.16b,%1.16b\n\t"\
    "mov %10.16b,%0.16b; mov %11.16b,%1.16b\n\t"\
    "mov %12.16b,%0.16b; mov %13.16b,%1.16b\n\t"\
    "mov %14.16b,%0.16b; mov %15.16b,%1.16b\n\t"\
    "mov %16.16b,%0.16b; mov %17.16b,%1.16b\n\t"\
    "mov %18.16b,%0.16b; mov %19.16b,%1.16b\n\t"\
    "mov %20.16b,%0.16b; mov %21.16b,%1.16b\n\t"\
    "mov %22.16b,%0.16b; mov %23.16b,%1.16b\n\t"\
    "cmp %w24,#0; b.eq 4f\n\t"\
    NEON_SGEMM_KERNEL_M##mdim##N##ndim##_PRELOAD_##cputype\
    "cmp %w24,#2; b.le 2f\n\t"\
    ".balign 16\n\t"\
    "1:\n\t"\
    NEON_SGEMM_KERNEL_M##mdim##N##ndim##_MAIN2_##cputype "b.gt 1b\n\t"\
    "2:\n\t"\
    "cmp %w24,#2; b.ne 3f\n\t"\
    NEON_SGEMM_KERNEL_M##mdim##N##ndim##_TAIL2_##cputype "b 4f\n\t"\
    "3:\n\t"\
    NEON_SGEMM_KERNEL_M##mdim##N##ndim##_TAIL1_##cputype\
    "4:\n\t"\
  :"=w"(cq01),"=w"(cq02),"=w"(cq03),"=w"(cq04),"=w"(cq05),"=w"(cq06),\
  "=w"(cq07),"=w"(cq08),"=w"(cq09),"=w"(cq10),"=w"(cq11),"=w"(cq12),\
  "=w"(cq13),"=w"(cq14),"=w"(cq15),"=w"(cq16),"=w"(cq17),"=w"(cq18),\
  "=w"(cq19),"=w"(cq20),"=w"(cq21),"=w"(cq22),"=w"(cq23),"=w"(cq24),\
  "+r"(k_left),"+r"(a_ptr),"+r"(b_ptr)\
  ::"cc","memory","x0","x1","v0","v1","v2","v3","v4","v5","v6","v7");\
  NEON_SGEMM_SAVE_M##mdim##N##ndim##_ASM1

#define NEON_SGEMM_KERNEL_M12N8_HALF_PRELOAD_A35 \
  "ld1r {v0.2s},[%25],#4\n\t"\
  "ldr d4,[%26]; ldr d5,[%26,#8]; ldr d6,[%26,#16]; add %26,%26,#32\n\t"

#define NEON_SGEMM_KERNEL_M12N8_HALF_MAIN2_A35 \
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %0.2s,v0.2s,v4.2s; fmla %1.2s,v0.2s,v5.2s; fmla %2.2s,v0.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %4.2s,v1.2s,v4.2s; fmla %5.2s,v1.2s,v5.2s; fmla %6.2s,v1.2s,v6.2s\n\t"\
  "ldr d7,[%26,#-8]\n\t"\
  "fmla %8.2s,v2.2s,v4.2s; fmla %9.2s,v2.2s,v5.2s; fmla %10.2s,v2.2s,v6.2s\n\t"\
  "ld1r {v3.2s},[%25],#4\n\t"\
  "fmla %3.2s,v0.2s,v7.2s; fmla %7.2s,v1.2s,v7.2s; fmla %11.2s,v2.2s,v7.2s\n\t"\
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %12.2s,v3.2s,v4.2s; fmla %13.2s,v3.2s,v5.2s; fmla %14.2s,v3.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %16.2s,v1.2s,v4.2s; add %25,%25,#24\n\t"\
  "fmla %17.2s,v1.2s,v5.2s; fmla %18.2s,v1.2s,v6.2s\n\t"\
  "ld1r {v0.2s},[%25],#4\n\t"\
  "fmla %20.2s,v2.2s,v4.2s; fmla %21.2s,v2.2s,v5.2s; fmla %22.2s,v2.2s,v6.2s\n\t"\
  "ldr d4,[%26]; ldr d5,[%26,#8]; ldr d6,[%26,#16]\n\t"\
  "fmla %15.2s,v3.2s,v7.2s; add %26,%26,#64\n\t"\
  "fmla %19.2s,v1.2s,v7.2s\n\t"\
  "fmla %23.2s,v2.2s,v7.2s\n\t"\
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %0.2s,v0.2s,v4.2s; fmla %1.2s,v0.2s,v5.2s; fmla %2.2s,v0.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %4.2s,v1.2s,v4.2s; fmla %5.2s,v1.2s,v5.2s; fmla %6.2s,v1.2s,v6.2s\n\t"\
  "ldr d7,[%26,#-40]\n\t"\
  "fmla %8.2s,v2.2s,v4.2s; fmla %9.2s,v2.2s,v5.2s; fmla %10.2s,v2.2s,v6.2s\n\t"\
  "ld1r {v3.2s},[%25],#4\n\t"\
  "fmla %3.2s,v0.2s,v7.2s; fmla %7.2s,v1.2s,v7.2s; fmla %11.2s,v2.2s,v7.2s\n\t"\
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %12.2s,v3.2s,v4.2s; fmla %13.2s,v3.2s,v5.2s; fmla %14.2s,v3.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %16.2s,v1.2s,v4.2s; add %25,%25,#24\n\t"\
  "fmla %17.2s,v1.2s,v5.2s; fmla %18.2s,v1.2s,v6.2s\n\t"\
  "ld1r {v0.2s},[%25],#4\n\t"\
  "fmla %20.2s,v2.2s,v4.2s; fmla %21.2s,v2.2s,v5.2s; fmla %22.2s,v2.2s,v6.2s\n\t"\
  "ldr d4,[%26,#-32]; ldr d5,[%26,#-24]; ldr d6,[%26,#-16]\n\t"\
  "fmla %15.2s,v3.2s,v7.2s; sub %w24,%w24,#2\n\t"\
  "fmla %19.2s,v1.2s,v7.2s; cmp %w24,#2\n\t"\
  "fmla %23.2s,v2.2s,v7.2s\n\t"

#define NEON_SGEMM_KERNEL_M12N8_HALF_TAIL2_A35 \
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %0.2s,v0.2s,v4.2s; fmla %1.2s,v0.2s,v5.2s; fmla %2.2s,v0.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %4.2s,v1.2s,v4.2s; fmla %5.2s,v1.2s,v5.2s; fmla %6.2s,v1.2s,v6.2s\n\t"\
  "ldr d7,[%26,#-8]\n\t"\
  "fmla %8.2s,v2.2s,v4.2s; fmla %9.2s,v2.2s,v5.2s; fmla %10.2s,v2.2s,v6.2s\n\t"\
  "ld1r {v3.2s},[%25],#4\n\t"\
  "fmla %3.2s,v0.2s,v7.2s; fmla %7.2s,v1.2s,v7.2s; fmla %11.2s,v2.2s,v7.2s\n\t"\
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %12.2s,v3.2s,v4.2s; fmla %13.2s,v3.2s,v5.2s; fmla %14.2s,v3.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %16.2s,v1.2s,v4.2s; add %25,%25,#24\n\t"\
  "fmla %17.2s,v1.2s,v5.2s; fmla %18.2s,v1.2s,v6.2s\n\t"\
  "ld1r {v0.2s},[%25],#4\n\t"\
  "fmla %20.2s,v2.2s,v4.2s; fmla %21.2s,v2.2s,v5.2s; fmla %22.2s,v2.2s,v6.2s\n\t"\
  "ldr d4,[%26]; ldr d5,[%26,#8]; ldr d6,[%26,#16]\n\t"\
  "fmla %15.2s,v3.2s,v7.2s; add %26,%26,#32\n\t"\
  "fmla %19.2s,v1.2s,v7.2s\n\t"\
  "fmla %23.2s,v2.2s,v7.2s\n\t"\
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %0.2s,v0.2s,v4.2s; fmla %1.2s,v0.2s,v5.2s; fmla %2.2s,v0.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %4.2s,v1.2s,v4.2s; fmla %5.2s,v1.2s,v5.2s; fmla %6.2s,v1.2s,v6.2s\n\t"\
  "ldr d7,[%26,#-8]\n\t"\
  "fmla %8.2s,v2.2s,v4.2s; fmla %9.2s,v2.2s,v5.2s; fmla %10.2s,v2.2s,v6.2s\n\t"\
  "ld1r {v3.2s},[%25],#4\n\t"\
  "fmla %3.2s,v0.2s,v7.2s; fmla %7.2s,v1.2s,v7.2s; fmla %11.2s,v2.2s,v7.2s\n\t"\
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %12.2s,v3.2s,v4.2s; fmla %13.2s,v3.2s,v5.2s; fmla %14.2s,v3.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %16.2s,v1.2s,v4.2s; fmla %17.2s,v1.2s,v5.2s; fmla %18.2s,v1.2s,v6.2s\n\t"\
  "fmla %20.2s,v2.2s,v4.2s; fmla %21.2s,v2.2s,v5.2s; fmla %22.2s,v2.2s,v6.2s\n\t"\
  "fmla %15.2s,v3.2s,v7.2s; add %25,%25,#24\n\t"\
  "fmla %19.2s,v1.2s,v7.2s\n\t"\
  "fmla %23.2s,v2.2s,v7.2s\n\t"

#define NEON_SGEMM_KERNEL_M12N8_HALF_TAIL1_A35 \
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %0.2s,v0.2s,v4.2s; fmla %1.2s,v0.2s,v5.2s; fmla %2.2s,v0.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %4.2s,v1.2s,v4.2s; fmla %5.2s,v1.2s,v5.2s; fmla %6.2s,v1.2s,v6.2s\n\t"\
  "ldr d7,[%26,#-8]\n\t"\
  "fmla %8.2s,v2.2s,v4.2s; fmla %9.2s,v2.2s,v5.2s; fmla %10.2s,v2.2s,v6.2s\n\t"\
  "ld1r {v3.2s},[%25],#4\n\t"\
  "fmla %3.2s,v0.2s,v7.2s; fmla %7.2s,v1.2s,v7.2s; fmla %11.2s,v2.2s,v7.2s\n\t"\
  "ld1r {v1.2s},[%25],#4\n\t"\
  "fmla %12.2s,v3.2s,v4.2s; fmla %13.2s,v3.2s,v5.2s; fmla %14.2s,v3.2s,v6.2s\n\t"\
  "ld1r {v2.2s},[%25],#4\n\t"\
  "fmla %16.2s,v1.2s,v4.2s; fmla %17.2s,v1.2s,v5.2s; fmla %18.2s,v1.2s,v6.2s\n\t"\
  "fmla %20.2s,v2.2s,v4.2s; fmla %21.2s,v2.2s,v5.2s; fmla %22.2s,v2.2s,v6.2s\n\t"\
  "fmla %15.2s,v3.2s,v7.2s; add %25,%25,#24\n\t"\
  "fmla %19.2s,v1.2s,v7.2s\n\t"\
  "fmla %23.2s,v2.2s,v7.2s\n\t"

#define NEON_SGEMM_SAVE_M6N2_UNIT_A35(c1, c2, c3, c4, c5, c6) \
  ct1 = vzip1_f32(c1, c2); ct2 = vzip1_f32(c3, c4); ct3 = vzip1_f32(c5, c6);\
  ct4 = vld1_f32(c_tmp), ct5 = vld1_f32(c_tmp + 2); ct6 = vld1_f32(c_tmp + 4);\
  ct1 = vfma_f32(ct1, ct4, beta_d);\
  ct2 = vfma_f32(ct2, ct5, beta_d);\
  ct3 = vfma_f32(ct3, ct6, beta_d);\
  vst1_f32(c_tmp, ct1); vst1_f32(c_tmp + 2, ct2); vst1_f32(c_tmp + 4, ct3);\
  c_tmp += ldc;\
  ct1 = vzip2_f32(c1, c2); ct2 = vzip2_f32(c3, c4); ct3 = vzip2_f32(c5, c6);\
  ct4 = vld1_f32(c_tmp), ct5 = vld1_f32(c_tmp + 2); ct6 = vld1_f32(c_tmp + 4);\
  ct1 = vfma_f32(ct1, ct4, beta_d);\
  ct2 = vfma_f32(ct2, ct5, beta_d);\
  ct3 = vfma_f32(ct3, ct6, beta_d);\
  vst1_f32(c_tmp, ct1); vst1_f32(c_tmp + 2, ct2); vst1_f32(c_tmp + 4, ct3);\
  c_tmp += ldc;

#define NEON_SGEMM_SAVE_M6N8_A35 \
  NEON_SGEMM_SAVE_M6N2_UNIT_A35(c01, c05, c09, c13, c17, c21)\
  NEON_SGEMM_SAVE_M6N2_UNIT_A35(c02, c06, c10, c14, c18, c22)\
  NEON_SGEMM_SAVE_M6N2_UNIT_A35(c03, c07, c11, c15, c19, c23)\
  NEON_SGEMM_SAVE_M6N2_UNIT_A35(c04, c08, c12, c16, c20, c24)

#define NEON_SGEMM_SAVE_M8N1_UNIT_A35(c1, c2, c3, c4) \
  ct1 = vld1_f32(c_tmp); ct2 = vld1_f32(c_tmp + 2);\
  ct3 = vld1_f32(c_tmp + 4); ct4 = vld1_f32(c_tmp + 6);\
  c1 = vfma_f32(c1, ct1, beta_d); c2 = vfma_f32(c2, ct2, beta_d);\
  c3 = vfma_f32(c3, ct3, beta_d); c4 = vfma_f32(c4, ct4, beta_d);\
  vst1_f32(c_tmp, c1); vst1_f32(c_tmp + 2, c2);\
  vst1_f32(c_tmp + 4, c3); vst1_f32(c_tmp + 6, c4); c_tmp += ldc;

#define NEON_SGEMM_SAVE_M8N6_A35 \
  NEON_SGEMM_SAVE_M8N1_UNIT_A35(c01, c02, c03, c04)\
  NEON_SGEMM_SAVE_M8N1_UNIT_A35(c05, c06, c07, c08)\
  NEON_SGEMM_SAVE_M8N1_UNIT_A35(c09, c10, c11, c12)\
  NEON_SGEMM_SAVE_M8N1_UNIT_A35(c13, c14, c15, c16)\
  NEON_SGEMM_SAVE_M8N1_UNIT_A35(c17, c18, c19, c20)\
  NEON_SGEMM_SAVE_M8N1_UNIT_A35(c21, c22, c23, c24)

#define NEON_SGEMM_KERNEL_M12N8_HALF_A35(a_ptr, b_ptr) \
  k_left = K;\
  __asm__ __volatile__ (\
    "movi %0.8b,#0; movi %1.8b,#0\n\t"\
    "mov %2.8b,%0.8b; mov %3.8b,%1.8b\n\t"\
    "mov %4.8b,%0.8b; mov %5.8b,%1.8b\n\t"\
    "mov %6.8b,%0.8b; mov %7.8b,%1.8b\n\t"\
    "mov %8.8b,%0.8b; mov %9.8b,%1.8b\n\t"\
    "mov %10.8b,%0.8b; mov %11.8b,%1.8b\n\t"\
    "mov %12.8b,%0.8b; mov %13.8b,%1.8b\n\t"\
    "mov %14.8b,%0.8b; mov %15.8b,%1.8b\n\t"\
    "mov %16.8b,%0.8b; mov %17.8b,%1.8b\n\t"\
    "mov %18.8b,%0.8b; mov %19.8b,%1.8b\n\t"\
    "mov %20.8b,%0.8b; mov %21.8b,%1.8b\n\t"\
    "mov %22.8b,%0.8b; mov %23.8b,%1.8b\n\t"\
    "cmp %w24,#0; b.eq 4f\n\t"\
    NEON_SGEMM_KERNEL_M12N8_HALF_PRELOAD_A35\
    "cmp %w24,#2; b.le 2f\n\t"\
    ".balign 16\n\t"\
    "1:\n\t"\
    NEON_SGEMM_KERNEL_M12N8_HALF_MAIN2_A35 "b.gt 1b\n\t"\
    "2:\n\t"\
    "cmp %w24,#2; b.ne 3f\n\t"\
    NEON_SGEMM_KERNEL_M12N8_HALF_TAIL2_A35 "b 4f\n\t"\
    "3:\n\t"\
    NEON_SGEMM_KERNEL_M12N8_HALF_TAIL1_A35\
    "4:\n\t"\
  :"=w"(c01),"=w"(c02),"=w"(c03),"=w"(c04),"=w"(c05),"=w"(c06),\
  "=w"(c07),"=w"(c08),"=w"(c09),"=w"(c10),"=w"(c11),"=w"(c12),\
  "=w"(c13),"=w"(c14),"=w"(c15),"=w"(c16),"=w"(c17),"=w"(c18),\
  "=w"(c19),"=w"(c20),"=w"(c21),"=w"(c22),"=w"(c23),"=w"(c24),\
  "+r"(k_left),"+r"(a_ptr),"+r"(b_ptr)\
  ::"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7");

#define NEON_SGEMM_COMPUTE_M8N12_A35 \
  uint32_t k_left;\
  float32x2_t c01, c02, c03, c04, c05, c06, c07, c08;\
  float32x2_t c09, c10, c11, c12, c13, c14, c15, c16;\
  float32x2_t c17, c18, c19, c20, c21, c22, c23, c24;\
  float *c_pref = c_ptr; PREF_C(8, 6)\
  const float *a_ptr = a_head;\
  const float *b_ptr = b_head;\
  NEON_SGEMM_KERNEL_M12N8_HALF_A35(b_ptr, a_ptr)\
  const float32x2_t beta_d = vdup_n_f32(beta);\
  float *c_tmp = c_ptr;\
  float32x2_t ct1, ct2, ct3, ct4;\
  NEON_SGEMM_SAVE_M8N6_A35\
  a_ptr = a_head; b_ptr = b_head + 6;\
  PREF_C(8, 6)\
  NEON_SGEMM_KERNEL_M12N8_HALF_A35(b_ptr, a_ptr)\
  NEON_SGEMM_SAVE_M8N6_A35

#define NEON_SGEMM_COMPUTE_M12N8_A35 \
  uint32_t k_left;\
  float32x2_t c01, c02, c03, c04, c05, c06, c07, c08;\
  float32x2_t c09, c10, c11, c12, c13, c14, c15, c16;\
  float32x2_t c17, c18, c19, c20, c21, c22, c23, c24;\
  float *c_pref = c_ptr; PREF_C(6, 8)\
  const float *a_ptr = a_head;\
  const float *b_ptr = b_head;\
  NEON_SGEMM_KERNEL_M12N8_HALF_A35(a_ptr, b_ptr)\
  const float32x2_t beta_d = vdup_n_f32(beta);\
  float *c_tmp = c_ptr;\
  float32x2_t ct1, ct2, ct3, ct4, ct5, ct6;\
  NEON_SGEMM_SAVE_M6N8_A35\
  c_tmp -= 8 * ldc;\
  c_tmp += 6;\
  c_pref = c_ptr + 6; PREF_C(6, 8)\
  b_ptr = b_head; a_ptr = a_head + 6;\
  NEON_SGEMM_KERNEL_M12N8_HALF_A35(a_ptr, b_ptr)\
  NEON_SGEMM_SAVE_M6N8_A35

#define CPUID_DETECT_MNK 1000000

void sgemm_kernel_lm_m8n12(uint32_t M, uint32_t N, uint32_t K, float beta,
  const float * __restrict__ sa, const float * __restrict__ sb,
  float * __restrict__ C, uint32_t ldc) {
  uint32_t n_left = N;
  const float *b_head = sb;
  float *c_head = C;
  uint32_t acc_mnk = CPUID_DETECT_MNK;
  uint8_t cpuid = 0, cputype = 0;
  for (; n_left > 11; n_left -= 12) {
    if (acc_mnk >= CPUID_DETECT_MNK) {
      cpuid = sched_getcpu();
      cputype = blas_arm_get_cpu_type(cpuid);
      acc_mnk = 0;
    }
    const float *a_head = sa;
    float *c_ptr = c_head;
    uint32_t m_left = M;
    if (cputype == 53) {
      for (; m_left > 7; m_left -= 8) {
        NEON_SGEMM_COMPUTE_ASM1(8, 12, A53)
        a_head += 8 * K;
        c_ptr += 8;
      }
    } else if (cputype == 55) {
      for (; m_left > 7; m_left -= 8) {
        NEON_SGEMM_COMPUTE_ASM1(8, 12, A55)
        a_head += 8 * K;
        c_ptr += 8;
      }
    } else if (cputype == 35) {
      for (; m_left > 7; m_left -= 8) {
        NEON_SGEMM_COMPUTE_M8N12_A35
        a_head += 8 * K;
        c_ptr += 8;
      }
    } else {
      for (; m_left > 7; m_left -= 8) {
        NEON_SGEMM_COMPUTE_ASM1(8, 12, A72)
        a_head += 8 * K;
        c_ptr += 8;
      }
    }
    MICRO_COMPUTE_LM(4, 12, float, float, float)
    b_head += K * 12;
    c_head += ldc * 12;
    acc_mnk += 12 * K * M;
  }
  ASSEMBLE_DUALPACK_COMPUTE_LM(8, float, float, float, 8)
}

void sgemm_kernel_ln_m12n8(uint32_t M, uint32_t N, uint32_t K, float beta,
  const float * __restrict__ sa, const float * __restrict__ sb,
  float * __restrict__ C, uint32_t ldc) {
  uint32_t m_left = M;
  const float *a_head = sa;
  float *c_head = C;
  uint32_t acc_mnk = CPUID_DETECT_MNK;
  uint8_t cpuid = 0, cputype = 0;
  for (; m_left > 11; m_left -= 12) {
    if (acc_mnk >= CPUID_DETECT_MNK) {
      cpuid = sched_getcpu();
      cputype = blas_arm_get_cpu_type(cpuid);
      acc_mnk = 0;
    }
    const float *b_head = sb;
    float *c_ptr = c_head;
    uint32_t n_left = N;
    if (cputype == 53) {
      for (; n_left > 7; n_left -= 8) {
        NEON_SGEMM_COMPUTE_ASM1(12, 8, A53)
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    } else if (cputype == 55) {
      for (; n_left > 7; n_left -= 8) {
        NEON_SGEMM_COMPUTE_ASM1(12, 8, A55)
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    } else if (cputype == 35) {
      for (; n_left > 7; n_left -= 8) {
        NEON_SGEMM_COMPUTE_M12N8_A35
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    } else {
      for (; n_left > 7; n_left -= 8) {
        NEON_SGEMM_COMPUTE_ASM1(12, 8, A72)
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    }
    MICRO_COMPUTE_LN(12, 4, float, float, float)
    a_head += K * 12;
    c_head += 12;
    acc_mnk += 12 * N * K;
  }
  ASSEMBLE_DUALPACK_COMPUTE_LN(8, float, float, float, 8)
}

