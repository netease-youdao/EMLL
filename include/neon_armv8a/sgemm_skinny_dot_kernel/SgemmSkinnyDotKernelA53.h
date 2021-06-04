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

#ifndef INCLUDE_A53_KERNEL
#define INCLUDE_A53_KERNEL

/* x0 - x3 for a_ptrs */
/* x4 for b_ptr, x5 for k_left */
/* x6 - x9 for a_pref */
/* x10 - x11, x16 - x17 and x19 - x20 for vec_fill */
/* x12 - x15 for c_tmp */

#define INIT_SAVE \
  "ldr s0,[%[beta_addr]]; mov x12,%[c_ptr]\n\t"\
  "add x13,%[c_ptr],%w[LDC],UXTW #2; add x14,%[c_ptr],%w[LDC],UXTW #3\n\t"\
  "add x15,x13,%w[LDC],UXTW #3\n\t"

#define UNIT_SAVE_M4N4_VR_CC(c1, c2, c3, c4) \
  "ldr q1,[x12]; ldr q2,[x13]; ldr q3,[x14]; ldr q4,[x15]\n\t"\
  "zip1 v5.4s,v"#c1".4s,v"#c2".4s; zip1 v6.4s,v"#c3".4s,v"#c4".4s\n\t"\
  "zip2 v7.4s,v"#c1".4s,v"#c2".4s; zip2 v"#c4".4s,v"#c3".4s,v"#c4".4s\n\t"\
  "zip1 v"#c1".2d,v5.2d,v6.2d; zip1 v"#c3".2d,v7.2d,v"#c4".2d\n\t"\
  "zip2 v"#c2".2d,v5.2d,v6.2d; zip2 v"#c4".2d,v7.2d,v"#c4".2d\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]; fmla v"#c2".4s,v2.4s,v0.s[0]\n\t"\
  "fmla v"#c3".4s,v3.4s,v0.s[0]; fmla v"#c4".4s,v4.4s,v0.s[0]\n\t"\
  "str q"#c1",[x12]; prfm pldl2keep,[x12,#32]\n\t"\
  "add x12,x12,%w[LDC],UXTW #4; prfm pstl1keep,[x12,#8]\n\t"\
  "str q"#c2",[x13]; prfm pldl2keep,[x13,#32]\n\t"\
  "add x13,x13,%w[LDC],UXTW #4; prfm pstl1keep,[x13,#8]\n\t"\
  "str q"#c3",[x14]; prfm pldl2keep,[x14,#32]\n\t"\
  "add x14,x14,%w[LDC],UXTW #4; prfm pstl1keep,[x14,#8]\n\t"\
  "str q"#c4",[x15]; prfm pldl2keep,[x15,#32]\n\t"\
  "add x15,x15,%w[LDC],UXTW #4; prfm pstl1keep,[x15,#8]\n\t"

#define UNIT_SAVE_M4N4_VR_CR(c1, c2, c3, c4) \
  "ldr q1,[x12]; ldr q2,[x13]; ldr q3,[x14]; ldr q4,[x15]\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]; fmla v"#c2".4s,v2.4s,v0.s[0]\n\t"\
  "fmla v"#c3".4s,v3.4s,v0.s[0]; fmla v"#c4".4s,v4.4s,v0.s[0]\n\t"\
  "str q"#c1",[x12],#16; str q"#c2",[x13],#16\n\t"\
  "str q"#c3",[x14],#16; str q"#c4",[x15],#16\n\t"

#define UNIT_SAVE_M4N4_VC_CC(c1, c2, c3, c4) \
  "ldr q1,[x12]; ldr q2,[x13]; ldr q3,[x14]; ldr q4,[x15]\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]; fmla v"#c2".4s,v2.4s,v0.s[0]\n\t"\
  "fmla v"#c3".4s,v3.4s,v0.s[0]; fmla v"#c4".4s,v4.4s,v0.s[0]\n\t"\
  "str q"#c1",[x12]; prfm pldl2keep,[x12,#32]\n\t"\
  "add x12,x12,%w[LDC],UXTW #4; prfm pstl1keep,[x12,#8]\n\t"\
  "str q"#c2",[x13]; prfm pldl2keep,[x13,#32]\n\t"\
  "add x13,x13,%w[LDC],UXTW #4; prfm pstl1keep,[x13,#8]\n\t"\
  "str q"#c3",[x14]; prfm pldl2keep,[x14,#32]\n\t"\
  "add x14,x14,%w[LDC],UXTW #4; prfm pstl1keep,[x14,#8]\n\t"\
  "str q"#c4",[x15]; prfm pldl2keep,[x15,#32]\n\t"\
  "add x15,x15,%w[LDC],UXTW #4; prfm pstl1keep,[x15,#8]\n\t"

#define UNIT_SAVE_M4N4_VC_CR(c1, c2, c3, c4) \
  "zip1 v1.4s,v"#c1".4s,v"#c2".4s; zip1 v2.4s,v"#c3".4s,v"#c4".4s\n\t"\
  "zip2 v3.4s,v"#c1".4s,v"#c2".4s; zip2 v4.4s,v"#c3".4s,v"#c4".4s\n\t"\
  "zip1 v"#c1".2d,v1.2d,v2.2d; zip2 v"#c2".2d,v1.2d,v2.2d\n\t"\
  "ldr q1,[x12]; ldr q2,[x13]\n\t"\
  "zip1 v"#c3".2d,v3.2d,v4.2d; zip2 v"#c4".2d,v3.2d,v4.2d\n\t"\
  "ldr q3,[x14]; ldr q4,[x15]\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]; fmla v"#c2".4s,v2.4s,v0.s[0]\n\t"\
  "fmla v"#c3".4s,v3.4s,v0.s[0]; fmla v"#c4".4s,v4.4s,v0.s[0]\n\t"\
  "str q"#c1",[x12],#16; str q"#c2",[x13],#16\n\t"\
  "str q"#c3",[x14],#16; str q"#c4",[x15],#16\n\t"

#define EDGE_SAVE_M4N1K4_CC(c1, c2, c3, c4) \
  "ldr q1,[x12]\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c2".4s\n\t"\
  "faddp v"#c3".4s,v"#c3".4s,v"#c4".4s\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c3".4s\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]; str q"#c1",[x12]\n\t"\
  "prfm pldl1keep,[x12,#32]; add x12,x12,%w[LDC],UXTW #2\n\t"

#define EDGE_SAVE_M4N1K4_CR(c1, c2, c3, c4) \
  "ldr s1,[x12]; ldr s2,[x13]; ldr s3,[x14]; ldr s4,[x15]\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c2".4s\n\t"\
  "ins v1.s[1],v2.s[0]; ins v3.s[1],v4.s[0]\n\t"\
  "faddp v"#c3".4s,v"#c3".4s,v"#c4".4s\n\t"\
  "ins v1.d[1],v3.d[0]\n\t"\
  "faddp v"#c1".4s,v"#c1".4s,v"#c3".4s\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]\n\t"\
  "st1 {v"#c1".s}[0],[x12],#4; st1 {v"#c1".s}[1],[x13],#4\n\t"\
  "st1 {v"#c1".s}[2],[x14],#4; st1 {v"#c1".s}[3],[x15],#4\n\t"

#define EDGE_SAVE_M4N1K2_CC(c1, c2) \
  "ldr q1,[x12]\n\t"\
  "trn1 v2.4s,v"#c1".4s,v"#c2".4s; trn2 v3.4s,v"#c1".4s,v"#c2".4s\n\t"\
  "fadd v2.4s,v2.4s,v3.4s; fmla v2.4s,v1.4s,v0.s[0]\n\t"\
  "str q2,[x12]; prfm pstl2keep,[x12,#32]; add x12,x12,%w[LDC],UXTW #2\n\t"

#define EDGE_SAVE_M4N1K2_CR(c1, c2) \
  "ldr s1,[x12]; ldr s2,[x13]; ldr s3,[x14]; ldr s4,[x15]\n\t"\
  "dup d5,v"#c1".d[1]; ins v1.s[1],v2.s[0]\n\t"\
  "dup d6,v"#c2".d[1]; ins v3.s[1],v4.s[0]\n\t"\
  "faddp v"#c1".2s,v"#c1".2s,v"#c2".2s; faddp v"#c2".2s,v5.2s,v6.2s\n\t"\
  "fmla v"#c1".2s,v1.2s,v0.s[0]; fmla v"#c2".2s,v3.2s,v0.s[0]\n\t"\
  "st1 {v"#c1".s}[0],[x12],#4; st1 {v"#c1".s}[1],[x13],#4\n\t"\
  "st1 {v"#c2".s}[0],[x14],#4; st1 {v"#c2".s}[1],[x15],#4\n\t"

#define EDGE_SAVE_M4N1K1_CC(c1) \
  "ldr q1,[x12]; fmla v"#c1".4s,v1.4s,v0.s[0]\n\t"\
  "str q"#c1",[x12]; prfm pstl2keep,[x12,#32]\n\t"\
  "add x12,x12,%w[LDC],UXTW #2\n\t"

#define EDGE_SAVE_M4N1K1_CR(c1) \
  "ldr s1,[x12]; ldr s2,[x13]; ldr s3,[x14]; ldr s4,[x15]\n\t"\
  "ins v1.s[1],v2.s[0]; ins v3.s[1],v4.s[0]; ins v1.d[1],v3.d[0]\n\t"\
  "fmla v"#c1".4s,v1.4s,v0.s[0]\n\t"\
  "st1 {v"#c1".s}[0],[x12],#4; st1 {v"#c1".s}[1],[x13],#4\n\t"\
  "st1 {v"#c1".s}[2],[x14],#4; st1 {v"#c1".s}[3],[x15],#4\n\t"

#define INIT_1V(c1) "movi v"#c1".16b,#0\n\t"

#define INIT_2V(c1, c2) \
  "movi v"#c1".16b,#0; movi v"#c2".16b,#0\n\t"\

#define INIT_4V(c1, c2, c3, c4) INIT_2V(c1, c2) INIT_2V(c3, c4)

/* m4n4 c_vec */
/* v28(v24) */
/* v29(v25) */
/* v30(v26) */
/* v31(v27) */
#define INIT_M4N4 \
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N4(mode) \
  "fadd v28.4s,v28.4s,v24.4s; fadd v29.4s,v29.4s,v25.4s\n\t"\
  "fadd v30.4s,v30.4s,v26.4s; fadd v31.4s,v31.4s,v27.4s\n\t"\
  UNIT_SAVE_M4N4_VR_##mode(28, 29, 30, 31)

#define KERNEL_M4N4_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#64; ldr d9,[x4,#-48]; ldr x10,[x4,#-40]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N4_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".s[0]; prfm pldl1keep,[x1,#80]\n\t"\
  "fmla v25.4s,v8.4s,v"#ac2".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-32]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[0]; ldr x10,[x4,#-24]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmov v10.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v28.4s,v9.4s,v"#ac1".s[1]; prfm pldl1keep,[x4,#96]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac2".s[1]; ldr x11,[x1,#-8]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d11,[x4,#-16]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-8]\n\t"\
  "fmla v31.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x2,#80]\n\t"\
  "fmov v11.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v24.4s,v10.4s,v"#ac1".s[2]; prfm pldl1keep,[x3,#80]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[2]; ldr x16,[x2,#-8]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d8,[x4]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[2]; ldr x10,[x4,#8]\n\t"\
  "fmla v27.4s,v10.4s,v"#ac4".s[2]; sub w5,w5,#4\n\t"\
  "fmov v8.d[1],x10; ldr d"#an4",[x3],#16\n\t"\
  "fmla v28.4s,v11.4s,v"#ac1".s[3]; cmp w5,#12\n\t"\
  "fmla v29.4s,v11.4s,v"#ac2".s[3]; ldr x11,[x3,#-8]\n\t"\
  "fmov v"#an3".d[1],x16; ldr d9,[x4,#16]\n\t"\
  "fmla v30.4s,v11.4s,v"#ac3".s[3]; ldr x10,[x4,#24]\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".s[3]; add x4,x4,#64\n\t"

#define KERNEL_M4N4_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v25.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-32]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[0]; ldr x10,[x4,#-24]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmov v10.d[1],x10\n\t"\
  "fmla v28.4s,v9.4s,v"#ac1".s[1]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "ldr d11,[x4,#-16]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-8]\n\t"\
  "fmla v31.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmov v11.d[1],x10\n\t"\
  "fmla v24.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[2]\n\t"\
  "fmla v27.4s,v10.4s,v"#ac4".s[2]; sub w5,w5,#4\n\t"\
  "fmla v28.4s,v11.4s,v"#ac1".s[3]; prfm pldl1keep,[x9]\n\t"\
  "fmla v29.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmla v30.4s,v11.4s,v"#ac3".s[3]\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".s[3]\n\t"

#define KERNEL_M4N4_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4\n\t"\
  "ldr q8,[x4],#16\n\t"\
  "ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "fmla v28.4s,v8.4s,v0.s[0]; sub w5,w5,#1\n\t"\
  "fmla v29.4s,v8.4s,v1.s[0]; cmp w5,#1\n\t"\
  "fmla v30.4s,v8.4s,v2.s[0]\n\t"\
  "fmla v31.4s,v8.4s,v3.s[0]\n\t"

/* m4n5 c_vec */
/* v21(v20) v22_comp */
/* v24(v23) v25_comp */
/* v27(v26) v28_comp */
/* v30(v29) v31_comp */

#define INIT_M4N5 \
  INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N5(mode) \
  "fadd v21.4s,v21.4s,v20.4s; fadd v24.4s,v24.4s,v23.4s\n\t"\
  "fadd v27.4s,v27.4s,v26.4s; fadd v30.4s,v30.4s,v29.4s\n\t"\
  UNIT_SAVE_M4N4_VR_##mode(21, 24, 27, 30) EDGE_SAVE_M4N1K4_##mode(22, 25, 28, 31)

#define KERNEL_M4N5_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#80; ldr d9,[x4,#-64]; ldr x10,[x4,#-56]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N5_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[0]; prfm pldl1keep,[x1,#80]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-48]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[0]; ldr x10,[x4,#-40]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmov v10.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[1]; prfm pldl1keep,[x4,#96]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[1]; ldr x11,[x1,#-8]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d11,[x4,#-32]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-24]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x2,#80]\n\t"\
  "fmov v11.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmla v23.4s,v10.4s,v"#ac2".s[2]; ldr x16,[x2,#-8]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d12,[x4,#-16]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[2]; ldr x10,[x4,#-8]\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[2]; sub w5,w5,#4\n\t"\
  "fmov v12.d[1],x10; ldr d"#an4",[x3],#16\n\t"\
  "fmla v21.4s,v11.4s,v"#ac1".s[3]; cmp w5,#12\n\t"\
  "fmla v24.4s,v11.4s,v"#ac2".s[3]; ldr x11,[x3,#-8]\n\t"\
  "fmov v"#an3".d[1],x16; ldr d8,[x4]\n\t"\
  "fmla v27.4s,v11.4s,v"#ac3".s[3]; ldr x10,[x4,#8]\n\t"\
  "fmla v30.4s,v11.4s,v"#ac4".s[3]; add x4,x4,#80\n\t"\
  "fmla v22.4s,v12.4s,v"#ac1".4s; prfm pldl1keep,[x3,#64]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-64]\n\t"\
  "fmla v25.4s,v12.4s,v"#ac2".4s; ldr x10,[x4,#-56]\n\t"\
  "fmla v28.4s,v12.4s,v"#ac3".4s; prfm pldl1keep,[x4,#48]\n\t"\
  "fmla v31.4s,v12.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N5_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-48]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[0]; ldr x10,[x4,#-40]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[1]; prfm pldl1keep,[x6]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "ldr d11,[x4,#-32]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-24]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmov v11.d[1],x10\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmla v23.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "ldr d12,[x4,#-16]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[2]; ldr x10,[x4,#-8]\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[2]; sub w5,w5,#4\n\t"\
  "fmov v12.d[1],x10\n\t"\
  "fmla v21.4s,v11.4s,v"#ac1".s[3]; prfm pldl1keep,[x8]\n\t"\
  "fmla v24.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmla v27.4s,v11.4s,v"#ac3".s[3]\n\t"\
  "fmla v30.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "fmla v22.4s,v12.4s,v"#ac1".4s\n\t"\
  "fmla v25.4s,v12.4s,v"#ac2".4s\n\t"\
  "fmla v28.4s,v12.4s,v"#ac3".4s; prfm pldl1keep,[x9]\n\t"\
  "fmla v31.4s,v12.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N5_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4\n\t"\
  "ldr q8,[x4],#16\n\t"\
  "ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "fmla v21.4s,v8.4s,v0.s[0]; sub w5,w5,#1\n\t"\
  "fmla v24.4s,v8.4s,v1.s[0]; cmp w5,#1\n\t"\
  "fmla v27.4s,v8.4s,v2.s[0]\n\t"\
  "ldr s9,[x4],#4\n\t"\
  "fmla v30.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v22.4s,v0.4s,v9.s[0]\n\t"\
  "fmla v25.4s,v1.4s,v9.s[0]\n\t"\
  "fmla v28.4s,v2.4s,v9.s[0]\n\t"\
  "fmla v31.4s,v3.4s,v9.s[0]\n\t"

/* m4n6 c_vec */
/* v17(v16) v18_comp v19_comp */
/* v21(v20) v22_comp v23_comp */
/* v25(v24) v26_comp v27_comp */
/* v29(v28) v30_comp v31_comp */

#define INIT_M4N6 \
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N6(mode) \
  "fadd v17.4s,v17.4s,v16.4s; fadd v21.4s,v21.4s,v20.4s\n\t"\
  "fadd v25.4s,v25.4s,v24.4s; fadd v29.4s,v29.4s,v28.4s\n\t"\
  UNIT_SAVE_M4N4_VR_##mode(17, 21, 25, 29) EDGE_SAVE_M4N1K4_##mode(18, 22, 26, 30)\
  EDGE_SAVE_M4N1K4_##mode(19, 23, 27, 31)

#define KERNEL_M4N6_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#96; ldr d9,[x4,#-80]; ldr x10,[x4,#-72]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N6_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]; prfm pldl1keep,[x1,#80]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-64]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[0]; ldr x10,[x4,#-56]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmov v10.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[1]; prfm pldl1keep,[x4,#96]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[1]; ldr x11,[x1,#-8]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d8,[x4,#-48]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x2,#80]\n\t"\
  "fmov v8.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".s[2]; ldr x16,[x2,#-8]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d9,[x4,#-32]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[2]; ldr x10,[x4,#-24]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac4".s[2]; sub w5,w5,#4\n\t"\
  "fmov v9.d[1],x10; ldr d10,[x4,#-16]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac1".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v25.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d"#an4",[x3],#16\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[3]; cmp w5,#12\n\t"\
  "fmla v18.4s,v9.4s,v"#ac1".4s; ldr x11,[x3,#-8]\n\t"\
  "fmla v22.4s,v9.4s,v"#ac2".4s; prfm pldl1keep,[x3,#64]\n\t"\
  "fmov v"#an3".d[1],x16; ldr d8,[x4]\n\t"\
  "fmla v26.4s,v9.4s,v"#ac3".4s; ldr x10,[x4,#8]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".4s; prfm pldl1keep,[x4,#144]\n\t"\
  "fmla v19.4s,v10.4s,v"#ac1".4s\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#16]\n\t"\
  "fmla v23.4s,v10.4s,v"#ac2".4s; ldr x10,[x4,#24]\n\t"\
  "fmla v27.4s,v10.4s,v"#ac3".4s; add x4,x4,#96\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N6_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-64]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[0]; ldr x10,[x4,#-56]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[1]; prfm pldl1keep,[x6]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "ldr d8,[x4,#-48]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmov v8.d[1],x10\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "ldr d9,[x4,#-32]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[2]; ldr x10,[x4,#-24]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac4".s[2]; sub w5,w5,#4\n\t"\
  "fmov v9.d[1],x10; ldr d10,[x4,#-16]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac1".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v25.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "fmov v10.d[1],x10\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[3]; prfm pldl1keep,[x8]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac1".4s\n\t"\
  "fmla v22.4s,v9.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v9.4s,v"#ac3".4s\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".4s; prfm pldl1keep,[x9]\n\t"\
  "fmla v19.4s,v10.4s,v"#ac1".4s\n\t"\
  "fmla v23.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v27.4s,v10.4s,v"#ac3".4s\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N6_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4\n\t"\
  "ldr q8,[x4],#16\n\t"\
  "ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "fmla v17.4s,v8.4s,v0.s[0]; sub w5,w5,#1\n\t"\
  "fmla v21.4s,v8.4s,v1.s[0]; cmp w5,#1\n\t"\
  "fmla v25.4s,v8.4s,v2.s[0]\n\t"\
  "ldr d9,[x4],#8\n\t"\
  "fmla v29.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v18.4s,v0.4s,v9.s[0]\n\t"\
  "fmla v22.4s,v1.4s,v9.s[0]\n\t"\
  "fmla v26.4s,v2.4s,v9.s[0]\n\t"\
  "fmla v30.4s,v3.4s,v9.s[0]\n\t"\
  "fmla v19.4s,v0.4s,v9.s[1]\n\t"\
  "fmla v23.4s,v1.4s,v9.s[1]\n\t"\
  "fmla v27.4s,v2.4s,v9.s[1]\n\t"\
  "fmla v31.4s,v3.4s,v9.s[1]\n\t"


/* m4n7 c_vec */
/* v13(v12) v14_comp v15_comp v16_comp */
/* v18(v17) v19_comp v20_comp v21_comp */
/* v23(v22) v24_comp v25_comp v26_comp */
/* v28(v27) v29_comp v30_comp v31_comp */

#define INIT_M4N7 \
  INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N7(mode) \
  "fadd v13.4s,v13.4s,v12.4s; fadd v18.4s,v18.4s,v17.4s\n\t"\
  "fadd v23.4s,v23.4s,v22.4s; fadd v28.4s,v28.4s,v27.4s\n\t"\
  UNIT_SAVE_M4N4_VR_##mode(13, 18, 23, 28) EDGE_SAVE_M4N1K4_##mode(14, 19, 24, 29)\
  EDGE_SAVE_M4N1K4_##mode(15, 20, 25, 30) EDGE_SAVE_M4N1K4_##mode(16, 21, 26, 31)

#define KERNEL_M4N7_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#112; ldr d9,[x4,#-96]; ldr x10,[x4,#-88]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N7_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-80]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmov v10.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[1]; prfm pldl1keep,[x4,#56]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[1]; ldr x11,[x1,#-8]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d8,[x4,#-64]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-56]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x1,#64]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-48]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v17.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[2]\n\t"\
  "fmla v27.4s,v10.4s,v"#ac4".s[2]; sub w5,w5,#4\n\t"\
  "fmla v13.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d10,[x4,#-32]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac2".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac3".s[3]; prfm pldl1keep,[x2,#64]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v14.4s,v9.4s,v"#ac1".4s; ldr x10,[x4,#-8]\n\t"\
  "fmla v19.4s,v9.4s,v"#ac2".4s; ldr x16,[x2,#-8]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac3".4s\n\t"\
  "fmov v11.d[1],x10; ldr d"#an4",[x3],#16\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".4s; cmp w5,#12\n\t"\
  "fmla v15.4s,v10.4s,v"#ac1".4s; prfm pldl1keep,[x3,#64]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmov v"#an3".d[1],x16; ldr d8,[x4]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac3".4s; ldr x10,[x4,#8]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".4s; prfm pldl1keep,[x4,#120]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".4s; ldr x11,[x3,#-8]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#16]\n\t"\
  "fmla v21.4s,v11.4s,v"#ac2".4s; ldr x10,[x4,#24]\n\t"\
  "fmla v26.4s,v11.4s,v"#ac3".4s; add x4,x4,#112\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N7_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-80]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "ldr d8,[x4,#-64]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-56]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-48]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v17.4s,v10.4s,v"#ac2".s[2]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[2]\n\t"\
  "fmla v27.4s,v10.4s,v"#ac4".s[2]; sub w5,w5,#4\n\t"\
  "fmla v13.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "ldr d10,[x4,#-32]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac2".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac3".s[3]; prfm pldl1keep,[x8]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v14.4s,v9.4s,v"#ac1".4s; ldr x10,[x4,#-8]\n\t"\
  "fmla v19.4s,v9.4s,v"#ac2".4s\n\t"\
  "fmla v24.4s,v9.4s,v"#ac3".4s\n\t"\
  "fmov v11.d[1],x10\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".4s\n\t"\
  "fmla v15.4s,v10.4s,v"#ac1".4s\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v25.4s,v10.4s,v"#ac3".4s\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".4s; prfm pldl1keep,[x9]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".4s\n\t"\
  "fmla v21.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v11.4s,v"#ac3".4s\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N7_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4\n\t"\
  "ldr q8,[x4],#16\n\t"\
  "ldr s2,[x2],#4; ldr s3,[x3],#4\n\t"\
  "fmla v13.4s,v8.4s,v0.s[0]; sub w5,w5,#1\n\t"\
  "fmla v18.4s,v8.4s,v1.s[0]; cmp w5,#1\n\t"\
  "fmla v23.4s,v8.4s,v2.s[0]\n\t"\
  "ldr d9,[x4],#8\n\t"\
  "fmla v28.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v14.4s,v0.4s,v9.s[0]\n\t"\
  "fmla v19.4s,v1.4s,v9.s[0]\n\t"\
  "ldr s10,[x4],#4\n\t"\
  "fmla v24.4s,v2.4s,v9.s[0]\n\t"\
  "fmla v29.4s,v3.4s,v9.s[0]\n\t"\
  "fmla v15.4s,v0.4s,v9.s[1]\n\t"\
  "fmla v20.4s,v1.4s,v9.s[1]\n\t"\
  "fmla v25.4s,v2.4s,v9.s[1]\n\t"\
  "fmla v30.4s,v3.4s,v9.s[1]\n\t"\
  "fmla v16.4s,v0.4s,v10.s[0]\n\t"\
  "fmla v21.4s,v1.4s,v10.s[0]\n\t"\
  "fmla v26.4s,v2.4s,v10.s[0]\n\t"\
  "fmla v31.4s,v3.4s,v10.s[0]\n\t"


/* m4n8 c_vec */
/* v24 - v25 */
/* v26 - v27 */
/* v28 - v29 */
/* v30 - v31 */

#define INIT_M4N8 \
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N8(mode) \
  UNIT_SAVE_M4N4_VR_##mode(24, 26, 28, 30) UNIT_SAVE_M4N4_VR_##mode(25, 27, 29, 31)

#define KERNEL_M4N8_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#128; ldr d9,[x4,#-112]; ldr x10,[x4,#-104]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N8_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac2".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-96]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac1".s[0]; prfm pldl1keep,[x4,#40]\n\t"\
  "fmla v30.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-80]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v31.4s,v9.4s,v"#ac4".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v11.d[1],x10; ldr d8,[x4,#-64]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-56]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[1]; sub w5,w5,#4\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v25.4s,v11.4s,v"#ac1".s[1]\n\t"\
  "fmla v27.4s,v11.4s,v"#ac2".s[1]; prfm pldl1keep,[x1,#64]\n\t"\
  "fmla v29.4s,v11.4s,v"#ac3".s[1]; cmp w5,#12\n\t"\
  "fmov v"#an1".d[1],x16; ldr d9,[x4,#-48]\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".s[2]; ldr x11,[x1,#-8]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac2".s[2]\n\t"\
  "fmov v9.d[1],x10; ldr d10,[x4,#-32]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac3".s[2]; ldr x10,[x4,#-24]\n\t"\
  "fmla v30.4s,v8.4s,v"#ac4".s[2]; prfm pldl1keep,[x4,#104]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac1".s[2]; prfm pldl1keep,[x3,#80]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac2".s[2]; ldr x10,[x4,#-8]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac3".s[2]; add x4,x4,#128\n\t"\
  "fmla v31.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "fmov v11.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v24.4s,v10.4s,v"#ac1".s[3]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac2".s[3]; prfm pldl1keep,[x2,#64]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[3]; ldr x16,[x2,#-8]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d8,[x4,#-128]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[3]; ldr x10,[x4,#-120]\n\t"\
  "fmla v25.4s,v11.4s,v"#ac1".s[3]; ldr x11,[x3],#16\n\t"\
  "fmla v27.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-112]\n\t"\
  "fmov v"#an3".d[1],x16; fmov d"#an4",x11\n\t"\
  "fmla v29.4s,v11.4s,v"#ac3".s[3]; ldr x10,[x4,#-104]\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".s[3]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N8_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-96]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac1".s[0]\n\t"\
  "fmla v30.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-80]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v31.4s,v9.4s,v"#ac4".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v11.d[1],x10; ldr d8,[x4,#-64]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-56]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[1]; sub w5,w5,#4\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v8.d[1],x10\n\t"\
  "fmla v25.4s,v11.4s,v"#ac1".s[1]\n\t"\
  "fmla v27.4s,v11.4s,v"#ac2".s[1]\n\t"\
  "fmla v29.4s,v11.4s,v"#ac3".s[1]\n\t"\
  "ldr d9,[x4,#-48]\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".s[2]; prfm pldl1keep,[x7]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac2".s[2]\n\t"\
  "fmov v9.d[1],x10; ldr d10,[x4,#-32]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac3".s[2]; ldr x10,[x4,#-24]\n\t"\
  "fmla v30.4s,v8.4s,v"#ac4".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac1".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac2".s[2]; ldr x10,[x4,#-8]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac3".s[2]\n\t"\
  "fmla v31.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "fmov v11.d[1],x10\n\t"\
  "fmla v24.4s,v10.4s,v"#ac1".s[3]; prfm pldl1keep,[x9]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac2".s[3]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[3]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[3]\n\t"\
  "fmla v25.4s,v11.4s,v"#ac1".s[3]\n\t"\
  "fmla v27.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmla v29.4s,v11.4s,v"#ac3".s[3]\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".s[3]\n\t"

#define KERNEL_M4N8_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; add x4,x4,#32\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v24.4s,v8.4s,v0.s[0]\n\t"\
  "fmla v25.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v26.4s,v8.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4\n\t"\
  "fmla v27.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v28.4s,v8.4s,v2.s[0]\n\t"\
  "fmla v29.4s,v9.4s,v2.s[0]; sub w5,w5,#1\n\t"\
  "fmla v30.4s,v8.4s,v3.s[0]; cmp w5,#1\n\t"\
  "fmla v31.4s,v9.4s,v3.s[0]\n\t"


/* m4n9 c_vec */
/* v20 - v21 v22_comp */
/* v23 - v24 v25_comp */
/* v26 - v27 v28_comp */
/* v29 - v30 v31_comp */

#define INIT_M4N9 \
  INIT_4V(20, 21, 22, 23) INIT_4V(24, 25, 26, 27)\
  INIT_4V(28, 29, 30, 31)

#define SAVE_M4N9(mode) \
  UNIT_SAVE_M4N4_VR_##mode(20, 23, 26, 29) UNIT_SAVE_M4N4_VR_##mode(21, 24, 27, 30)\
  EDGE_SAVE_M4N1K4_##mode(22, 25, 28, 31)

#define KERNEL_M4N9_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#144; ldr d9,[x4,#-128]; ldr x10,[x4,#-120]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N9_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-112]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[0]; prfm pldl1keep,[x4,#24]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-96]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-80]\n\t"\
  "fmla v23.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-72]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[1]; sub w5,w5,#4\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v21.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac2".s[1]; prfm pldl1keep,[x1,#64]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d10,[x4,#-64]\n\t"\
  "fmla v30.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-56]\n\t"\
  "fmla v20.4s,v9.4s,v"#ac1".s[2]; ldr x11,[x1,#-8]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-48]\n\t"\
  "fmla v26.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[2]; prfm pldl1keep,[x4,#88]\n\t"\
  "fmla v21.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-32]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-24]\n\t"\
  "fmla v27.4s,v10.4s,v"#ac3".s[2]; add x4,x4,#144\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[3]; prfm pldl1keep,[x2,#64]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[3]; cmp w5,#12\n\t"\
  "fmov v"#an2".d[1],x11; ldr d10,[x4,#-160]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-152]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[3]; prfm pldl1keep,[x4,#8]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[3]; ldr x16,[x2,#-8]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-144]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[3]; ldr x10,[x4,#-136]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[3]; ldr x11,[x3],#16\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".4s\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-128]\n\t"\
  "fmov v"#an3".d[1],x16; fmov d"#an4",x11\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".4s; ldr x10,[x4,#-120]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".4s; ldr x11,[x3,#-8]\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".4s; prfm pldl1keep,[x3,#64]\n\t"

#define KERNEL_M4N9_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-112]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[0]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-96]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-80]\n\t"\
  "fmla v23.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-72]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[1]; sub w5,w5,#4\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v21.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "ldr d10,[x4,#-64]\n\t"\
  "fmla v30.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-56]\n\t"\
  "fmla v20.4s,v9.4s,v"#ac1".s[2]; prfm pldl1keep,[x7]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-48]\n\t"\
  "fmla v26.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v21.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-32]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-24]\n\t"\
  "fmla v27.4s,v10.4s,v"#ac3".s[2]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "ldr d10,[x4,#-16]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[3]; prfm pldl1keep,[x9]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "fmov v10.d[1],x10\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[3]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".4s\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".4s\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N9_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; add x4,x4,#36\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v20.4s,v8.4s,v0.s[0]\n\t"\
  "fmla v21.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v23.4s,v8.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4\n\t"\
  "fmla v24.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v26.4s,v8.4s,v2.s[0]\n\t"\
  "fmla v27.4s,v9.4s,v2.s[0]; sub w5,w5,#1\n\t"\
  "ldr s10,[x4,#-4]\n\t"\
  "fmla v29.4s,v8.4s,v3.s[0]; cmp w5,#1\n\t"\
  "fmla v30.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v22.4s,v10.4s,v0.4s\n\t"\
  "fmla v25.4s,v10.4s,v1.4s\n\t"\
  "fmla v28.4s,v10.4s,v2.4s\n\t"\
  "fmla v31.4s,v10.4s,v3.4s\n\t"


/* m4n10 c_vec */
/* v16 - v17 v18_comp v19_comp */
/* v20 - v21 v22_comp v23_comp */
/* v24 - v25 v26_comp v27_comp */
/* v28 - v29 v30_comp v31_comp */

#define INIT_M4N10 \
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N10(mode) \
  UNIT_SAVE_M4N4_VR_##mode(16, 20, 24, 28) UNIT_SAVE_M4N4_VR_##mode(17, 21, 25, 29)\
  EDGE_SAVE_M4N1K4_##mode(18, 22, 26, 30) EDGE_SAVE_M4N1K4_##mode(19, 23, 27, 31)

#define KERNEL_M4N10_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#160; ldr d9,[x4,#-144]; ldr x10,[x4,#-136]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N10_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-128]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[0]; prfm pldl1keep,[x4,#32]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-112]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-96]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[1]; sub w5,w5,#4\n\t"\
  "fmla v28.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v17.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[1]; prfm pldl1keep,[x1,#64]\n\t"\
  "fmla v25.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d10,[x4,#-80]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-72]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[2]; ldr x11,[x1,#-8]\n\t"\
  "fmla v20.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-64]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[2]; prfm pldl1keep,[x4,#96]\n\t"\
  "fmla v17.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-48]\n\t"\
  "fmla v21.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac3".s[2]; add x4,x4,#160\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[3]; cmp w5,#12\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[3]; prfm pldl1keep,[x2,#64]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d10,[x4,#-192]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-184]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[3]; ldr x16,[x2,#-8]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-176]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[3]; ldr x10,[x4,#-168]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[3]; ldr x11,[x3],#16\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".4s\n\t"\
  "fmov v11.d[1],x10; ldr d8,[x4,#-160]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".4s; ldr x10,[x4,#-152]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".4s; prfm pldl1keep,[x4]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".4s\n\t"\
  "fmov v"#an3".d[1],x16; fmov d"#an4",x11\n\t"\
  "fmla v19.4s,v11.4s,v"#ac1".4s; ldr x11,[x3,#-8]\n\t"\
  "fmla v23.4s,v11.4s,v"#ac2".4s; prfm pldl1keep,[x3,#64]\n\t"\
  "fmla v27.4s,v11.4s,v"#ac3".4s\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-144]\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s; ldr x10,[x4,#-136]\n\t"

#define KERNEL_M4N10_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-128]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[0]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-112]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v16.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-96]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[1]; sub w5,w5,#4\n\t"\
  "fmla v28.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v17.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v25.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "ldr d10,[x4,#-80]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-72]\n\t"\
  "fmla v16.4s,v9.4s,v"#ac1".s[2]; prfm pldl1keep,[x7]\n\t"\
  "fmla v20.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-64]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v17.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-48]\n\t"\
  "fmla v21.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac3".s[2]\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "ldr d10,[x4,#-32]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[3]; prfm pldl1keep,[x9]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".4s\n\t"\
  "fmov v11.d[1],x10\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".4s\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".4s\n\t"\
  "fmla v19.4s,v11.4s,v"#ac1".4s\n\t"\
  "fmla v23.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmla v27.4s,v11.4s,v"#ac3".4s\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N10_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; add x4,x4,#40\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v16.4s,v8.4s,v0.s[0]\n\t"\
  "fmla v17.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v20.4s,v8.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4\n\t"\
  "fmla v21.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v24.4s,v8.4s,v2.s[0]\n\t"\
  "fmla v25.4s,v9.4s,v2.s[0]; sub w5,w5,#1\n\t"\
  "ldr d10,[x4,#-8]\n\t"\
  "fmla v28.4s,v8.4s,v3.s[0]; cmp w5,#1\n\t"\
  "fmla v29.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v18.4s,v0.4s,v10.s[0]\n\t"\
  "fmla v22.4s,v1.4s,v10.s[0]\n\t"\
  "fmla v26.4s,v2.4s,v10.s[0]\n\t"\
  "fmla v30.4s,v3.4s,v10.s[0]\n\t"\
  "fmla v19.4s,v0.4s,v10.s[1]\n\t"\
  "fmla v23.4s,v1.4s,v10.s[1]\n\t"\
  "fmla v27.4s,v2.4s,v10.s[1]\n\t"\
  "fmla v31.4s,v3.4s,v10.s[1]\n\t"


/* m4n11 c_vec */
/* v12 - v13 v14_comp v15_comp v16_comp */
/* v17 - v18 v19_comp v20_comp v21_comp */
/* v22 - v23 v24_comp v25_comp v26_comp */
/* v27 - v28 v29_comp v30_comp v31_comp */

#define INIT_M4N11 \
  INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23) INIT_4V(24, 25, 26, 27)\
  INIT_4V(28, 29, 30, 31)

#define SAVE_M4N11(mode) \
  UNIT_SAVE_M4N4_VR_##mode(12, 17, 22, 27) UNIT_SAVE_M4N4_VR_##mode(13, 18, 23, 28)\
  EDGE_SAVE_M4N1K4_##mode(14, 19, 24, 29) EDGE_SAVE_M4N1K4_##mode(15, 20, 25, 30)\
  EDGE_SAVE_M4N1K4_##mode(16, 21, 26, 31)

#define KERNEL_M4N11_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#176; ldr d9,[x4,#-160]; ldr x10,[x4,#-152]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N11_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-144]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-136]\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[0]; prfm pldl1keep,[x4,#48]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-128]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-112]\n\t"\
  "fmla v17.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-104]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[1]; sub w5,w5,#4\n\t"\
  "fmla v27.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v13.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac2".s[1]; prfm pldl1keep,[x1,#64]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d10,[x4,#-96]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v12.4s,v9.4s,v"#ac1".s[2]; ldr x11,[x1,#-8]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-80]\n\t"\
  "fmla v22.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac4".s[2]; prfm pldl1keep,[x4,#112]\n\t"\
  "fmla v13.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v11.d[1],x10; ldr d8,[x4,#-64]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v23.4s,v10.4s,v"#ac3".s[2]; add x4,x4,#176\n\t"\
  "fmla v28.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v12.4s,v11.4s,v"#ac1".s[3]; cmp w5,#12\n\t"\
  "fmla v17.4s,v11.4s,v"#ac2".s[3]; prfm pldl1keep,[x2,#64]\n\t"\
  "fmla v22.4s,v11.4s,v"#ac3".s[3]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d9,[x4,#-224]\n\t"\
  "fmla v27.4s,v11.4s,v"#ac4".s[3]; ldr x10,[x4,#-216]\n\t"\
  "fmla v13.4s,v8.4s,v"#ac1".s[3]; prfm pldl1keep,[x4]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmov v9.d[1],x10; ldr d10,[x4,#-208]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac3".s[3]; ldr x10,[x4,#-200]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[3]; ldr x16,[x2,#-8]\n\t"\
  "fmla v14.4s,v9.4s,v"#ac1".4s\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-192]\n\t"\
  "fmla v19.4s,v9.4s,v"#ac2".4s; ldr x10,[x4,#-184]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac3".4s\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".4s\n\t"\
  "fmov v11.d[1],x10; ldr d8,[x4,#-176]\n\t"\
  "fmla v15.4s,v10.4s,v"#ac1".4s; ldr x10,[x4,#-168]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v25.4s,v10.4s,v"#ac3".4s\n\t"\
  "fmov v"#an3".d[1],x16; ldr d"#an4",[x3],#16\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".4s\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".4s; prfm pldl1keep,[x3,#64]\n\t"\
  "fmla v21.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-160]\n\t"\
  "fmla v26.4s,v11.4s,v"#ac3".4s; ldr x10,[x4,#-152]\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N11_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-144]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-136]\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[0]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-128]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v12.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-112]\n\t"\
  "fmla v17.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-104]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac3".s[1]; sub w5,w5,#4\n\t"\
  "fmla v27.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v13.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "ldr d10,[x4,#-96]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v12.4s,v9.4s,v"#ac1".s[2]; prfm pldl1keep,[x7]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-80]\n\t"\
  "fmla v22.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac4".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v13.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v11.d[1],x10; ldr d8,[x4,#-64]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v23.4s,v10.4s,v"#ac3".s[2]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v8.d[1],x10\n\t"\
  "fmla v12.4s,v11.4s,v"#ac1".s[3]\n\t"\
  "fmla v17.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmla v22.4s,v11.4s,v"#ac3".s[3]\n\t"\
  "ldr d9,[x4,#-48]\n\t"\
  "fmla v27.4s,v11.4s,v"#ac4".s[3]; ldr x10,[x4,#-40]\n\t"\
  "fmla v13.4s,v8.4s,v"#ac1".s[3]; prfm pldl1keep,[x9]\n\t"\
  "fmla v18.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmov v9.d[1],x10; ldr d10,[x4,#-32]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac3".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[3]\n\t"\
  "fmla v14.4s,v9.4s,v"#ac1".4s\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v19.4s,v9.4s,v"#ac2".4s; ldr x10,[x4,#-8]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac3".4s\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".4s\n\t"\
  "fmov v11.d[1],x10\n\t"\
  "fmla v15.4s,v10.4s,v"#ac1".4s\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v25.4s,v10.4s,v"#ac3".4s\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".4s\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".4s\n\t"\
  "fmla v21.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v11.4s,v"#ac3".4s\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N11_TL1 \
  "ldr s0,[x0],#4; ldr s1,[x1],#4\n\t"\
  "ldr q8,[x4]; ldr q9,[x4,#16]; add x4,x4,#44\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]\n\t"\
  "fmla v13.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v17.4s,v8.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4\n\t"\
  "fmla v18.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v22.4s,v8.4s,v2.s[0]\n\t"\
  "fmla v23.4s,v9.4s,v2.s[0]; sub w5,w5,#1\n\t"\
  "ldr d10,[x4,#-12]\n\t"\
  "fmla v27.4s,v8.4s,v3.s[0]; cmp w5,#1\n\t"\
  "fmla v28.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v14.4s,v0.4s,v10.s[0]\n\t"\
  "ldr s11,[x4,#-4]\n\t"\
  "fmla v19.4s,v1.4s,v10.s[0]\n\t"\
  "fmla v24.4s,v2.4s,v10.s[0]\n\t"\
  "fmla v29.4s,v3.4s,v10.s[0]\n\t"\
  "fmla v15.4s,v0.4s,v10.s[1]\n\t"\
  "fmla v20.4s,v1.4s,v10.s[1]\n\t"\
  "fmla v25.4s,v2.4s,v10.s[1]\n\t"\
  "fmla v30.4s,v3.4s,v10.s[1]\n\t"\
  "fmla v16.4s,v0.4s,v11.s[0]\n\t"\
  "fmla v21.4s,v1.4s,v11.s[0]\n\t"\
  "fmla v26.4s,v2.4s,v11.s[0]\n\t"\
  "fmla v31.4s,v3.4s,v11.s[0]\n\t"


/* m4n12 c_vec */
/* v20 - v22 */
/* v23 - v25 */
/* v26 - v28 */
/* v29 - v31 */

#define INIT_M4N12 \
  INIT_4V(20, 21, 22, 23) INIT_4V(24, 25, 26, 27)\
  INIT_4V(28, 29, 30, 31)

#define SAVE_M4N12(mode) \
  UNIT_SAVE_M4N4_VR_##mode(20, 23, 26, 29) UNIT_SAVE_M4N4_VR_##mode(21, 24, 27, 30)\
  UNIT_SAVE_M4N4_VR_##mode(22, 25, 28, 31)

#define KERNEL_M4N12_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#192; ldr d9,[x4,#-176]; ldr x10,[x4,#-168]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N12_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-160]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-152]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[0]; prfm pldl1keep,[x4,#8]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-144]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-136]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".s[0]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-128]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".s[0]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[1]; prfm pldl1keep,[x1,#64]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d10,[x4,#-112]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-104]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[1]; ldr x11,[x1,#-8]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-96]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x4,#72]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-80]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-72]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[1]\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[2]; prfm pldl1keep,[x2,#64]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[2]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d10,[x4,#-64]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[2]; ldr x16,[x2,#-8]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-48]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-32]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-24]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[2]; prfm pldl1keep,[x4,#136]\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an4",[x3],#16\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[3]; sub w5,w5,#4\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[3]; prfm pldl1keep,[x3,#64]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "fmov v"#an3".d[1],x16; ldr d10,[x4,#-16]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[3]; ldr x11,[x3,#-8]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[3]; ldr x10,[x4,#8]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[3]; cmp w5,#12\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".s[3]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#16]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[3]; ldr x10,[x4,#24]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[3]; add x4,x4,#192\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".s[3]\n\t"

#define KERNEL_M4N12_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v"#ac4".d[1],x11; fmov v9.d[1],x10\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "ldr d10,[x4,#-160]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[0]; ldr x10,[x4,#-152]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[0]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-144]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-136]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".s[0]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-128]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[0]\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".s[0]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "ldr d10,[x4,#-112]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-104]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[1]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-96]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-80]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-72]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[1]\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[2]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[2]\n\t"\
  "ldr d10,[x4,#-64]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[2]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-48]\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-32]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-24]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[2]\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v20.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v23.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "ldr d10,[x4,#-16]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac1".s[3]; sub w5,w5,#4\n\t"\
  "fmla v24.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "fmov v10.d[1],x10\n\t"\
  "fmla v27.4s,v9.4s,v"#ac3".s[3]\n\t"\
  "fmla v30.4s,v9.4s,v"#ac4".s[3]; prfm pldl1keep,[x9]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac1".s[3]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac2".s[3]\n\t"\
  "fmla v28.4s,v10.4s,v"#ac3".s[3]\n\t"\
  "fmla v31.4s,v10.4s,v"#ac4".s[3]\n\t"

#define KERNEL_M4N12_TL1 \
  "ldr s0,[x0],#4; ldr q8,[x4]; ldr q9,[x4,#16]\n\t"\
  "ldr q10,[x4,#32]; add x4,x4,#48\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v20.4s,v8.4s,v0.s[0]\n\t"\
  "fmla v21.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v22.4s,v10.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v23.4s,v8.4s,v1.s[0]\n\t"\
  "fmla v24.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v25.4s,v10.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4\n\t"\
  "fmla v26.4s,v8.4s,v2.s[0]; sub w5,w5,#1\n\t"\
  "fmla v27.4s,v9.4s,v2.s[0]\n\t"\
  "fmla v28.4s,v10.4s,v2.s[0]\n\t"\
  "cmp w5,#1\n\t"\
  "fmla v29.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v30.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v31.4s,v10.4s,v3.s[0]\n\t"


/* m4n13 c_vec */
/* v16 - v18 v19_comp */
/* v20 - v22 v23_comp */
/* v24 - v26 v27_comp */
/* v28 - v30 v31_comp */

#define INIT_M4N13 \
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N13(mode) \
  UNIT_SAVE_M4N4_VR_##mode(16, 20, 24, 28) UNIT_SAVE_M4N4_VR_##mode(17, 21, 25, 29)\
  UNIT_SAVE_M4N4_VR_##mode(18, 22, 26, 30) EDGE_SAVE_M4N1K4_##mode(19, 23, 27, 31)

#define KERNEL_M4N13_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#208; ldr d9,[x4,#-192]; ldr x10,[x4,#-184]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N13_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-176]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-168]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[0]; prfm pldl1keep,[x4,#24]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-160]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-152]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".s[0]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-144]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".s[0]; ldr x10,[x4,#-136]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[0]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[1]; prfm pldl1keep,[x1,#64]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d10,[x4,#-128]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-120]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[1]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-112]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-104]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x4,#88]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-96]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[1]; ldr x11,[x1,#-8]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[2]; prfm pldl1keep,[x2,#64]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[2]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d10,[x4,#-80]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[2]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-64]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[2]; ldr x16,[x2,#-8]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-48]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[2]; prfm pldl1keep,[x4,#152]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an4",[x3],#16\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[3]; prfm pldl1keep,[x3,#64]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "fmov v"#an3".d[1],x16; ldr d10,[x4,#-32]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[3]; sub w5,w5,#4\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[3]; cmp w5,#12\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".s[3]\n\t"\
  "fmov v11.d[1],x10; ldr d8,[x4]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".s[3]; ldr x16,[x4,#8]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[3]; ldr x11,[x3,#-8]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[3]\n\t"\
  "ldr d9,[x4,#16]\n\t"\
  "fmla v19.4s,v11.4s,v"#ac1".4s; ldr x10,[x4,#24]\n\t"\
  "fmla v23.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmov v8.d[1],x16\n\t"\
  "fmla v27.4s,v11.4s,v"#ac3".4s; add x4,x4,#208\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N13_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-176]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-168]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[0]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-160]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-152]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".s[0]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-144]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".s[0]; ldr x10,[x4,#-136]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[0]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[0]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "ldr d10,[x4,#-128]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-120]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[1]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-112]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-104]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-96]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[1]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[2]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[2]\n\t"\
  "ldr d10,[x4,#-80]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[2]\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-64]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-48]\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[2]; prfm pldl1keep,[x9]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v20.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v24.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "ldr d10,[x4,#-32]\n\t"\
  "fmla v28.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v17.4s,v9.4s,v"#ac1".s[3]; sub w5,w5,#4\n\t"\
  "fmla v21.4s,v9.4s,v"#ac2".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac3".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "fmla v18.4s,v10.4s,v"#ac1".s[3]\n\t"\
  "fmov v11.d[1],x10\n\t"\
  "fmla v22.4s,v10.4s,v"#ac2".s[3]\n\t"\
  "fmla v26.4s,v10.4s,v"#ac3".s[3]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".s[3]\n\t"\
  "fmla v19.4s,v11.4s,v"#ac1".4s\n\t"\
  "fmla v23.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmla v27.4s,v11.4s,v"#ac3".4s\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N13_TL1 \
  "ldr s0,[x0],#4; ldr q8,[x4]; ldr q9,[x4,#16]\n\t"\
  "ldr q10,[x4,#32]; add x4,x4,#52\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v16.4s,v8.4s,v0.s[0]\n\t"\
  "fmla v17.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v18.4s,v10.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v20.4s,v8.4s,v1.s[0]\n\t"\
  "fmla v21.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v22.4s,v10.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4\n\t"\
  "fmla v24.4s,v8.4s,v2.s[0]; sub w5,w5,#1\n\t"\
  "fmla v25.4s,v9.4s,v2.s[0]\n\t"\
  "fmla v26.4s,v10.4s,v2.s[0]\n\t"\
  "ldr s11,[x4,#-4]\n\t"\
  "fmla v28.4s,v8.4s,v3.s[0]; cmp w5,#1\n\t"\
  "fmla v29.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v30.4s,v10.4s,v3.s[0]\n\t"\
  "fmla v19.4s,v0.4s,v11.4s\n\t"\
  "fmla v23.4s,v1.4s,v11.4s\n\t"\
  "fmla v27.4s,v2.4s,v11.4s\n\t"\
  "fmla v31.4s,v3.4s,v11.4s\n\t"


/* m4n14 c_vec */
/* v12 - v14 v15_comp v16_comp */
/* v17 - v19 v20_comp v21_comp */
/* v22 - v24 v25_comp v26_comp */
/* v27 - v29 v30_comp v31_comp */

#define INIT_M4N14 \
  INIT_4V(12, 13, 14, 15) INIT_4V(16, 17, 18, 19)\
  INIT_4V(20, 21, 22, 23) INIT_4V(24, 25, 26, 27)\
  INIT_4V(28, 29, 30, 31)

#define SAVE_M4N14(mode) \
  UNIT_SAVE_M4N4_VR_##mode(12, 17, 22, 27) UNIT_SAVE_M4N4_VR_##mode(13, 18, 23, 28)\
  UNIT_SAVE_M4N4_VR_##mode(14, 19, 24, 29) EDGE_SAVE_M4N1K4_##mode(15, 20, 25, 30)\
  EDGE_SAVE_M4N1K4_##mode(16, 21, 26, 31)

#define KERNEL_M4N14_PRELOAD4 \
  "ldr q0,[x0],#16; ldr q1,[x1],#16; ldr q2,[x2],#16; ldr d3,[x3],#16\n\t"\
  "ldr q8,[x4],#224; ldr d9,[x4,#-208]; ldr x10,[x4,#-200]; ldr x11,[x3,#-8]\n\t"

#define KERNEL_M4N14_K8_L4(ac1, ac2, ac3, ac4, an1, an2, an3, an4) \
  "fmov v9.d[1],x10; ldr d"#an1",[x0],#16\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[0]; prfm pldl1keep,[x0,#64]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-192]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-184]\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[0]; prfm pldl1keep,[x4,#8]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-176]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-168]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[0]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac1".s[0]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-160]\n\t"\
  "fmla v19.4s,v10.4s,v"#ac2".s[0]; ldr x10,[x4,#-152]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[0]; ldr x16,[x0,#-8]\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[0]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an2",[x1],#16\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[1]; prfm pldl1keep,[x1,#64]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d10,[x4,#-144]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-136]\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[1]; ldr x11,[x1,#-8]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-128]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-120]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x4,#72]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-112]\n\t"\
  "fmla v19.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-104]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[1]\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10; ldr d"#an3",[x2],#16\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[2]; prfm pldl1keep,[x2,#64]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[2]\n\t"\
  "fmov v"#an2".d[1],x11; ldr d10,[x4,#-96]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[2]; ldr x10,[x4,#-88]\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[2]; ldr x16,[x2,#-8]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-80]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d11,[x4,#-64]\n\t"\
  "fmla v19.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[2]; prfm pldl1keep,[x4,#136]\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v11.d[1],x10; ldr d"#an4",[x3],#16\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[3]; prfm pldl1keep,[x3,#64]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "fmov v"#an3".d[1],x16; ldr d9,[x4,#-48]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-40]\n\t"\
  "fmla v13.4s,v11.4s,v"#ac1".s[3]; sub w5,w5,#4\n\t"\
  "fmla v18.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmov v9.d[1],x10; ldr d10,[x4,#-32]\n\t"\
  "fmla v23.4s,v11.4s,v"#ac3".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v28.4s,v11.4s,v"#ac4".s[3]; cmp w5,#12\n\t"\
  "fmla v14.4s,v9.4s,v"#ac1".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v19.4s,v9.4s,v"#ac2".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac3".s[3]; ldr x11,[x3,#-8]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "fmov v11.d[1],x10; ldr d8,[x4]\n\t"\
  "fmla v15.4s,v10.4s,v"#ac1".4s; ldr x16,[x4,#8]\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".4s\n\t"\
  "fmla v25.4s,v10.4s,v"#ac3".4s\n\t"\
  "ldr d9,[x4,#16]\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".4s; ldr x10,[x4,#24]\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".4s; add x4,x4,#224\n\t"\
  "fmla v21.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmov v8.d[1],x16\n\t"\
  "fmla v26.4s,v11.4s,v"#ac3".4s\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N14_K8_T4(ac1, ac2, ac3, ac4) \
  "fmov v9.d[1],x10\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[0]\n\t"\
  "fmov v"#ac4".d[1],x11; ldr d10,[x4,#-192]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[0]; ldr x10,[x4,#-184]\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[0]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[0]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-176]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[0]; ldr x10,[x4,#-168]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac1".s[0]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-160]\n\t"\
  "fmla v19.4s,v10.4s,v"#ac2".s[0]; ldr x10,[x4,#-152]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[0]\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[0]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[1]\n\t"\
  "ldr d10,[x4,#-144]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[1]; ldr x10,[x4,#-136]\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[1]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[1]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-128]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[1]; ldr x10,[x4,#-120]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac1".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-112]\n\t"\
  "fmla v19.4s,v10.4s,v"#ac2".s[1]; ldr x10,[x4,#-104]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[1]\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[1]\n\t"\
  "fmov v9.d[1],x10\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[2]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[2]\n\t"\
  "ldr d10,[x4,#-96]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[2]; ldr x10,[x4,#-88]\n\t"\
  "fmla v13.4s,v9.4s,v"#ac1".s[2]\n\t"\
  "fmla v18.4s,v9.4s,v"#ac2".s[2]\n\t"\
  "fmov v10.d[1],x10; ldr d8,[x4,#-80]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac3".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v28.4s,v9.4s,v"#ac4".s[2]\n\t"\
  "fmla v14.4s,v10.4s,v"#ac1".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d11,[x4,#-64]\n\t"\
  "fmla v19.4s,v10.4s,v"#ac2".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v24.4s,v10.4s,v"#ac3".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v29.4s,v10.4s,v"#ac4".s[2]\n\t"\
  "fmov v11.d[1],x10\n\t"\
  "fmla v12.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v17.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v22.4s,v8.4s,v"#ac3".s[3]\n\t"\
  "ldr d9,[x4,#-48]\n\t"\
  "fmla v27.4s,v8.4s,v"#ac4".s[3]; ldr x10,[x4,#-40]\n\t"\
  "fmla v13.4s,v11.4s,v"#ac1".s[3]; sub w5,w5,#4\n\t"\
  "fmla v18.4s,v11.4s,v"#ac2".s[3]\n\t"\
  "fmov v9.d[1],x10; ldr d10,[x4,#-32]\n\t"\
  "fmla v23.4s,v11.4s,v"#ac3".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v28.4s,v11.4s,v"#ac4".s[3]\n\t"\
  "fmla v14.4s,v9.4s,v"#ac1".s[3]\n\t"\
  "fmov v10.d[1],x10; ldr d11,[x4,#-16]\n\t"\
  "fmla v19.4s,v9.4s,v"#ac2".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v24.4s,v9.4s,v"#ac3".s[3]\n\t"\
  "fmla v29.4s,v9.4s,v"#ac4".s[3]\n\t"\
  "fmov v11.d[1],x10\n\t"\
  "fmla v15.4s,v10.4s,v"#ac1".4s\n\t"\
  "fmla v20.4s,v10.4s,v"#ac2".4s; prfm pldl1keep,[x9]\n\t"\
  "fmla v25.4s,v10.4s,v"#ac3".4s\n\t"\
  "fmla v30.4s,v10.4s,v"#ac4".4s\n\t"\
  "fmla v16.4s,v11.4s,v"#ac1".4s\n\t"\
  "fmla v21.4s,v11.4s,v"#ac2".4s\n\t"\
  "fmla v26.4s,v11.4s,v"#ac3".4s\n\t"\
  "fmla v31.4s,v11.4s,v"#ac4".4s\n\t"

#define KERNEL_M4N14_TL1 \
  "ldr s0,[x0],#4; ldr q8,[x4]; ldr q9,[x4,#16]\n\t"\
  "ldr q10,[x4,#32]; add x4,x4,#56\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v12.4s,v8.4s,v0.s[0]\n\t"\
  "fmla v13.4s,v9.4s,v0.s[0]\n\t"\
  "fmla v14.4s,v10.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v17.4s,v8.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v9.4s,v1.s[0]\n\t"\
  "fmla v19.4s,v10.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4\n\t"\
  "fmla v22.4s,v8.4s,v2.s[0]; sub w5,w5,#1\n\t"\
  "fmla v23.4s,v9.4s,v2.s[0]\n\t"\
  "fmla v24.4s,v10.4s,v2.s[0]\n\t"\
  "ldr d11,[x4,#-8]\n\t"\
  "fmla v27.4s,v8.4s,v3.s[0]; cmp w5,#1\n\t"\
  "fmla v28.4s,v9.4s,v3.s[0]\n\t"\
  "fmla v29.4s,v10.4s,v3.s[0]\n\t"\
  "fmla v15.4s,v0.4s,v11.s[0]\n\t"\
  "fmla v20.4s,v1.4s,v11.s[0]\n\t"\
  "fmla v25.4s,v2.4s,v11.s[0]\n\t"\
  "fmla v30.4s,v3.4s,v11.s[0]\n\t"\
  "fmla v16.4s,v0.4s,v11.s[1]\n\t"\
  "fmla v21.4s,v1.4s,v11.s[1]\n\t"\
  "fmla v26.4s,v2.4s,v11.s[1]\n\t"\
  "fmla v31.4s,v3.4s,v11.s[1]\n\t"

#define FUNC_K4(ndim) \
static inline void sgemm_skinny1_a53_m4n##ndim(\
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
    "cmp w5,#12; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    KERNEL_M4N##ndim##_K8_L4(0, 1, 2, 3, 4, 5, 6, 7)\
    KERNEL_M4N##ndim##_K8_L4(4, 5, 6, 7, 0, 1, 2, 3)\
    "b.ge 1b; 2:\n\t"\
    "cmp w5,#8; b.lt 3f\n\t"\
    KERNEL_M4N##ndim##_K8_L4(0, 1, 2, 3, 4, 5, 6, 7)\
    KERNEL_M4N##ndim##_K8_T4(4, 5, 6, 7)\
    "b 4f; 3:\n\t"\
    KERNEL_M4N##ndim##_K8_T4(0, 1, 2, 3)\
    "4:\n\t"\
    "cmp w5,#1; b.lt 6f\n\t"\
    "5:\n\t"\
    KERNEL_M4N##ndim##_TL1\
    "b.ge 5b; 6:\n\t"\
    INIT_SAVE\
    "cmp %w[c_rowmajor],#0; b.eq 7f\n\t"\
    SAVE_M4N##ndim(CR) "b 8f\n\t"\
    "7:\n\t"\
    SAVE_M4N##ndim(CC)\
    "8:\n\t"\
  ::[a_ptr]"r"(a_ptr), [c_ptr]"r"(c_ptr), [b_scr]"r"(b_scr),\
    [K]"r"(K), [LDA]"r"(LDA), [LDC]"r"(LDC),\
    [beta_addr]"r"(beta_addr), [c_rowmajor]"r"(c_rowmajor)\
  :"cc","memory","x0","x1","x2","x3","x4","x5","x6","x7","x8","x9",\
  "x10","x11","x12","x13","x14","x15","x16",\
  "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13",\
  "v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25",\
  "v26","v27","v28","v29","v30","v31");\
}

FUNC_K4(4)
FUNC_K4(5)
FUNC_K4(6)
FUNC_K4(7)
FUNC_K4(8)
FUNC_K4(9)
FUNC_K4(10)
FUNC_K4(11)
FUNC_K4(12)
FUNC_K4(13)
FUNC_K4(14)

/* m4n15 c_vec */
/* v14 - v16 v23_comp v24_comp v25_comp */
/* v17 - v19 v29_comp v30_comp v31_comp */
/* v20 - v22 v23_comp v24_comp v25_comp */
/* v26 - v28 v29_comp v30_comp v31_comp */

#define INIT_M4N15 \
  INIT_4V(14, 15, 16, 17) INIT_4V(18, 19, 20, 21) INIT_4V(22, 23, 24, 25)\
  INIT_4V(26, 27, 28, 29) INIT_2V(30, 31)

#define SAVE_M4N15(mode) \
  UNIT_SAVE_M4N4_VR_##mode(14, 17, 20, 26) UNIT_SAVE_M4N4_VR_##mode(15, 18, 21, 27)\
  UNIT_SAVE_M4N4_VR_##mode(16, 19, 22, 28) EDGE_SAVE_M4N1K2_##mode(23, 29)\
  EDGE_SAVE_M4N1K2_##mode(24, 30) EDGE_SAVE_M4N1K2_##mode(25, 31)

#define KERNEL_M4N15_PRELOAD2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr x16,[x2],#8; ldr x11,[x3],#8\n\t"\
  "ldr q4,[x4]; ldr d5,[x4,#16]; ldr x10,[x4,#24]\n\t"\
  "add x4,x4,#120; fmov v0.d[1],x16\n\t"

#define KERNEL_M4N15_MAIN2(ac1, ac2, an1, an2, ap1, ap2) \
  "fmov v5.d[1],x10; ldr d"#an1",[x0],#8\n\t"\
  "fmla v14.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-88]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-80]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#64]\n\t"\
  "fmla v26.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d4,[x4,#-72]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-64]\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[2]; ldr x16,[x2],#8\n\t"\
  "fmla v16.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-56]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-48]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac1".s[2]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v5.d[1],x10; ldr d"#an2",[x1],#8\n\t"\
  "fmla v14.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[1]; prfm pldl1keep,[x4,#128]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac1".s[3]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d6,[x4,#-40]\n\t"\
  "fmla v26.4s,v4.4s,v"#ac2".s[3]; ldr x10,[x4,#-32]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-24]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac1".s[3]; ldr x10,[x4,#-16]\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[3]; ldr x11,[x3],#8\n\t"\
  "fmla v16.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "ins v7.d[1],v7.d[0]; dup v8.2d,x10\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[1]; ldr x10,[x4,#-8]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac1".s[3]; add x4,x4,#120\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "dup v6.2d,x10; ldr d4,[x4,#-120]\n\t"\
  "fmla v23.4s,v7.4s,v"#ac1".4s; ldr x10,[x4,#-112]\n\t"\
  "fmla v29.4s,v7.4s,v"#ac2".4s; sub w5,w5,#2\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".4s\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-104]\n\t"\
  "fmla v30.4s,v8.4s,v"#ac2".4s; ldr x10,[x4,#-96]\n\t"\
  "fmla v25.4s,v6.4s,v"#ac1".4s; cmp w5,#6\n\t"\
  "fmla v31.4s,v6.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N15_TAIL2(ac1, ac2) \
  "fmov v5.d[1],x10\n\t"\
  "fmla v14.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[0]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-88]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-80]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[0]\n\t"\
  "fmla v26.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d4,[x4,#-72]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-64]\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v16.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-56]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-48]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac1".s[2]; prfm pldl1keep,[x6]\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v5.d[1],x10\n\t"\
  "fmla v14.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac1".s[3]\n\t"\
  "ldr d6,[x4,#-40]\n\t"\
  "fmla v26.4s,v4.4s,v"#ac2".s[3]; ldr x10,[x4,#-32]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-24]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac1".s[3]; ldr x10,[x4,#-16]\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmla v16.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "ins v7.d[1],v7.d[0]; dup v8.2d,x10\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[1]; ldr x10,[x4,#-8]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "dup v6.2d,x10\n\t"\
  "fmla v23.4s,v7.4s,v"#ac1".4s; prfm pldl1keep,[x8]\n\t"\
  "fmla v29.4s,v7.4s,v"#ac2".4s; sub w5,w5,#2\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".4s\n\t"\
  "fmla v30.4s,v8.4s,v"#ac2".4s; prfm pldl1keep,[x9]\n\t"\
  "fmla v25.4s,v6.4s,v"#ac1".4s\n\t"\
  "fmla v31.4s,v6.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N15_FIN1 \
  "ldr s0,[x0],#4; ldr q4,[x4]; ldr q5,[x4,#16]\n\t"\
  "ldr q6,[x4,#32]; add x4,x4,#60\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v14.4s,v4.4s,v0.s[0]\n\t"\
  "fmla v15.4s,v5.4s,v0.s[0]\n\t"\
  "fmla v16.4s,v6.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v17.4s,v4.4s,v1.s[0]; ldr w10,[x4,#-12]\n\t"\
  "fmla v18.4s,v5.4s,v1.s[0]\n\t"\
  "fmla v19.4s,v6.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4; dup v7.2d,x10\n\t"\
  "fmla v20.4s,v4.4s,v2.s[0]; ldr w11,[x4,#-8]\n\t"\
  "fmla v21.4s,v5.4s,v2.s[0]\n\t"\
  "fmla v22.4s,v6.4s,v2.s[0]\n\t"\
  "ins v0.d[1],v2.d[0]; dup v8.2d,x11\n\t"\
  "fmla v26.4s,v4.4s,v3.s[0]; ldr w16,[x4,#-4]\n\t"\
  "fmla v27.4s,v5.4s,v3.s[0]\n\t"\
  "fmla v28.4s,v6.4s,v3.s[0]\n\t"\
  "ins v1.d[1],v3.d[0]; dup v6.2d,x16\n\t"\
  "fmla v23.4s,v7.4s,v0.4s\n\t"\
  "fmla v24.4s,v8.4s,v0.4s\n\t"\
  "fmla v29.4s,v7.4s,v1.4s\n\t"\
  "fmla v30.4s,v8.4s,v1.4s\n\t"\
  "fmla v25.4s,v6.4s,v0.4s\n\t"\
  "fmla v31.4s,v6.4s,v1.4s\n\t"


/* m4n16 c_vec */
/* v16 - v19 */
/* v20 - v23 */
/* v24 - v27 */
/* v28 - v31 */

#define INIT_M4N16 \
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N16(mode) \
  UNIT_SAVE_M4N4_VR_##mode(16, 20, 24, 28) UNIT_SAVE_M4N4_VR_##mode(17, 21, 25, 29)\
  UNIT_SAVE_M4N4_VR_##mode(18, 22, 26, 30) UNIT_SAVE_M4N4_VR_##mode(19, 23, 27, 31)

#define KERNEL_M4N16_PRELOAD2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr x16,[x2],#8; ldr x11,[x3],#8\n\t"\
  "ldr q4,[x4]; ldr d5,[x4,#16]; ldr x10,[x4,#24]\n\t"\
  "add x4,x4,#128; fmov v0.d[1],x16\n\t"

#define KERNEL_M4N16_MAIN2(ac1, ac2, an1, an2, ap1, ap2) \
  "fmov v5.d[1],x10; ldr d"#an1",[x0],#8\n\t"\
  "fmla v16.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v24.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-96]\n\t"\
  "fmla v17.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#64]\n\t"\
  "fmla v28.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-80]\n\t"\
  "fmla v25.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v29.4s,v5.4s,v"#ac2".s[2]; ldr x16,[x2],#8\n\t"\
  "fmla v18.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-64]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-56]\n\t"\
  "fmla v26.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v30.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d"#an2",[x1],#8\n\t"\
  "fmla v19.4s,v7.4s,v"#ac1".s[0]\n\t"\
  "fmla v23.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v27.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d5,[x4,#-48]\n\t"\
  "fmla v31.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v16.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac2".s[1]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-32]\n\t"\
  "fmla v24.4s,v4.4s,v"#ac1".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v28.4s,v4.4s,v"#ac2".s[3]; ldr x11,[x3],#8\n\t"\
  "fmla v17.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-16]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-8]\n\t"\
  "fmla v25.4s,v5.4s,v"#ac1".s[3]; add x4,x4,#128\n\t"\
  "fmla v29.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-128]\n\t"\
  "fmla v18.4s,v6.4s,v"#ac1".s[1]; ldr x16,[x4,#-120]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x4]\n\t"\
  "fmla v26.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "ldr d5,[x4,#-112]\n\t"\
  "fmla v30.4s,v6.4s,v"#ac2".s[3]; ldr x10,[x4,#-104]\n\t"\
  "fmla v19.4s,v7.4s,v"#ac1".s[1]; sub w5,w5,#2\n\t"\
  "fmla v23.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "fmov v4.d[1],x16\n\t"\
  "fmla v27.4s,v7.4s,v"#ac1".s[3]; cmp w5,#6\n\t"\
  "fmla v31.4s,v7.4s,v"#ac2".s[3]\n\t"

#define KERNEL_M4N16_TAIL2(ac1, ac2) \
  "fmov v5.d[1],x10\n\t"\
  "fmla v16.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac2".s[0]\n\t"\
  "fmla v24.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-96]\n\t"\
  "fmla v17.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v28.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-80]\n\t"\
  "fmla v25.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v29.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v18.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-64]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-56]\n\t"\
  "fmla v26.4s,v6.4s,v"#ac1".s[2]; prfm pldl1keep,[x7]\n\t"\
  "fmla v30.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v19.4s,v7.4s,v"#ac1".s[0]\n\t"\
  "fmla v23.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x8]\n\t"\
  "fmla v27.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "ldr d5,[x4,#-48]\n\t"\
  "fmla v31.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v16.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac2".s[1]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-32]\n\t"\
  "fmla v24.4s,v4.4s,v"#ac1".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v28.4s,v4.4s,v"#ac2".s[3]\n\t"\
  "fmla v17.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-16]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-8]\n\t"\
  "fmla v25.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmla v29.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10\n\t"\
  "fmla v18.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x9]\n\t"\
  "fmla v26.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "fmla v30.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "fmla v19.4s,v7.4s,v"#ac1".s[1]; sub w5,w5,#2\n\t"\
  "fmla v23.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "fmla v27.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v31.4s,v7.4s,v"#ac2".s[3]\n\t"

#define KERNEL_M4N16_FIN1 \
  "ldr s0,[x0],#4; ldr q4,[x4]; ldr q5,[x4,#16]\n\t"\
  "ldr q6,[x4,#32]; add x4,x4,#64\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v16.4s,v4.4s,v0.s[0]\n\t"\
  "fmla v17.4s,v5.4s,v0.s[0]\n\t"\
  "fmla v18.4s,v6.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v20.4s,v4.4s,v1.s[0]\n\t"\
  "fmla v21.4s,v5.4s,v1.s[0]\n\t"\
  "fmla v22.4s,v6.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4; ldr d7,[x4,#-16]\n\t"\
  "fmla v24.4s,v4.4s,v2.s[0]; ldr x10,[x4,#-8]\n\t"\
  "fmla v25.4s,v5.4s,v2.s[0]\n\t"\
  "fmla v26.4s,v6.4s,v2.s[0]\n\t"\
  "fmov v7.d[1],x10\n\t"\
  "fmla v28.4s,v4.4s,v3.s[0]\n\t"\
  "fmla v29.4s,v5.4s,v3.s[0]\n\t"\
  "fmla v30.4s,v6.4s,v3.s[0]\n\t"\
  "fmla v19.4s,v7.4s,v0.s[0]\n\t"\
  "fmla v23.4s,v7.4s,v1.s[0]\n\t"\
  "fmla v27.4s,v7.4s,v2.s[0]\n\t"\
  "fmla v31.4s,v7.4s,v3.s[0]\n\t"


/* m4n17 c_vec */
/* v14 - v17 v26_comp */
/* v18 - v21 v31_comp */
/* v22 - v25 v26_comp */
/* v27 - v30 v31_comp */

#define INIT_M4N17 INIT_2V(14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N17(mode) \
  UNIT_SAVE_M4N4_VR_##mode(14, 18, 22, 27) UNIT_SAVE_M4N4_VR_##mode(15, 19, 23, 28)\
  UNIT_SAVE_M4N4_VR_##mode(16, 20, 24, 29) UNIT_SAVE_M4N4_VR_##mode(17, 21, 25, 30)\
  EDGE_SAVE_M4N1K2_##mode(26, 31)

#define KERNEL_M4N17_PRELOAD2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr x16,[x2],#8; ldr x11,[x3],#8\n\t"\
  "ldr q4,[x4]; ldr d5,[x4,#16]; ldr x10,[x4,#24]\n\t"\
  "add x4,x4,#136; fmov v0.d[1],x16\n\t"

#define KERNEL_M4N17_MAIN2(ac1, ac2, an1, an2, ap1, ap2) \
  "fmov v5.d[1],x10; ldr d"#an1",[x0],#8\n\t"\
  "fmla v14.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v18.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-104]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-96]\n\t"\
  "fmla v19.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#64]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-88]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-80]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[2]; ldr x16,[x2],#8\n\t"\
  "fmla v16.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-72]\n\t"\
  "fmla v20.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-64]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d"#an2",[x1],#8\n\t"\
  "fmla v17.4s,v7.4s,v"#ac1".s[0]\n\t"\
  "fmla v21.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d5,[x4,#-56]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-48]\n\t"\
  "fmla v14.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmla v18.4s,v4.4s,v"#ac2".s[1]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-40]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[3]; ldr x10,[x4,#-32]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[3]; prfm pldl1keep,[x4,#112]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-24]\n\t"\
  "fmla v19.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-16]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[3]; ldr x11,[x3],#8\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-8]\n\t"\
  "fmla v16.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v20.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x4,#160]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "ins v8.d[1],v8.d[0]; ldr d4,[x4]\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[3]; ldr x16,[x4,#8]\n\t"\
  "fmla v17.4s,v7.4s,v"#ac1".s[1]; add x4,x4,#136\n\t"\
  "fmla v21.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "ldr d5,[x4,#-120]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[3]; ldr x10,[x4,#-112]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[3]; sub w5,w5,#2\n\t"\
  "fmov v4.d[1],x16\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".4s; cmp w5,#6\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N17_TAIL2(ac1, ac2) \
  "fmov v5.d[1],x10\n\t"\
  "fmla v14.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v18.4s,v4.4s,v"#ac2".s[0]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-104]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-96]\n\t"\
  "fmla v19.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-88]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-80]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v16.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-72]\n\t"\
  "fmla v20.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-64]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v17.4s,v7.4s,v"#ac1".s[0]\n\t"\
  "fmla v21.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x7]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "ldr d5,[x4,#-56]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-48]\n\t"\
  "fmla v14.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmla v18.4s,v4.4s,v"#ac2".s[1]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-40]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[3]; ldr x10,[x4,#-32]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[3]; prfm pldl1keep,[x8]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-24]\n\t"\
  "fmla v19.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-16]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-8]\n\t"\
  "fmla v16.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v20.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x9]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "ins v8.d[1],v8.d[0]\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "fmla v17.4s,v7.4s,v"#ac1".s[1]\n\t"\
  "fmla v21.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[3]; sub w5,w5,#2\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".4s\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N17_FIN1 \
  "ldr s0,[x0],#4; ldr q4,[x4]; ldr q5,[x4,#16]\n\t"\
  "ldr q6,[x4,#32]; add x4,x4,#68\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v14.4s,v4.4s,v0.s[0]\n\t"\
  "fmla v15.4s,v5.4s,v0.s[0]\n\t"\
  "fmla v16.4s,v6.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v18.4s,v4.4s,v1.s[0]\n\t"\
  "fmla v19.4s,v5.4s,v1.s[0]\n\t"\
  "fmla v20.4s,v6.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4; ldr d7,[x4,#-20]\n\t"\
  "fmla v22.4s,v4.4s,v2.s[0]; ldr x10,[x4,#-12]\n\t"\
  "fmla v23.4s,v5.4s,v2.s[0]\n\t"\
  "fmla v24.4s,v6.4s,v2.s[0]\n\t"\
  "fmov v7.d[1],x10; ldr s8,[x4,#-4]\n\t"\
  "fmla v27.4s,v4.4s,v3.s[0]\n\t"\
  "fmla v28.4s,v5.4s,v3.s[0]\n\t"\
  "fmla v29.4s,v6.4s,v3.s[0]\n\t"\
  "ins v8.d[1],v8.d[0]\n\t"\
  "fmla v17.4s,v7.4s,v0.s[0]\n\t"\
  "fmla v21.4s,v7.4s,v1.s[0]\n\t"\
  "fmla v25.4s,v7.4s,v2.s[0]\n\t"\
  "ins v0.d[1],v2.d[0]; ins v1.d[1],v3.d[0]\n\t"\
  "fmla v30.4s,v7.4s,v3.s[0]\n\t"\
  "fmla v26.4s,v8.4s,v0.4s\n\t"\
  "fmla v31.4s,v8.4s,v1.4s\n\t"


/* m4n18 c_vec */
/* v12 - v15 v24_comp v25_comp */
/* v16 - v19 v30_comp v31_comp */
/* v20 - v23 v24_comp v25_comp */
/* v26 - v29 v30_comp v31_comp */

#define INIT_M4N18 INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N18(mode) \
  UNIT_SAVE_M4N4_VR_##mode(12, 16, 20, 26) UNIT_SAVE_M4N4_VR_##mode(13, 17, 21, 27)\
  UNIT_SAVE_M4N4_VR_##mode(14, 18, 22, 28) UNIT_SAVE_M4N4_VR_##mode(15, 19, 23, 29)\
  EDGE_SAVE_M4N1K2_##mode(24, 30) EDGE_SAVE_M4N1K2_##mode(25, 31)

#define KERNEL_M4N18_PRELOAD2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr x16,[x2],#8; ldr x11,[x3],#8\n\t"\
  "ldr q4,[x4]; ldr d5,[x4,#16]; ldr x10,[x4,#24]\n\t"\
  "add x4,x4,#144; fmov v0.d[1],x16\n\t"

#define KERNEL_M4N18_MAIN2(ac1, ac2, an1, an2, ap1, ap2) \
  "fmov v5.d[1],x10; ldr d"#an1",[x0],#8\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v16.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-112]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v17.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#64]\n\t"\
  "fmla v26.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-96]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-88]\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[2]; ldr x16,[x2],#8\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-80]\n\t"\
  "fmla v18.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d"#an2",[x1],#8\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[0]\n\t"\
  "fmla v19.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v23.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d5,[x4,#-64]\n\t"\
  "fmla v29.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmla v16.4s,v4.4s,v"#ac2".s[1]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-48]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac1".s[3]; ldr x10,[x4,#-40]\n\t"\
  "fmla v26.4s,v4.4s,v"#ac2".s[3]; prfm pldl1keep,[x4,#112]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-32]\n\t"\
  "fmla v17.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-24]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac1".s[3]; ldr x11,[x3],#8\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-16]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v18.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x4,#160]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "ins v8.d[1],v8.d[0]; ldr d9,[x4,#-8]\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[1]; add x4,x4,#144\n\t"\
  "fmla v19.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "ins v9.d[1],v9.d[0]; ldr d4,[x4,#-144]\n\t"\
  "fmla v23.4s,v7.4s,v"#ac1".s[3]; ldr x10,[x4,#-136]\n\t"\
  "fmla v29.4s,v7.4s,v"#ac2".s[3]; sub w5,w5,#2\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".4s\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-128]\n\t"\
  "fmla v30.4s,v8.4s,v"#ac2".4s; ldr x10,[x4,#-120]\n\t"\
  "fmla v25.4s,v9.4s,v"#ac1".4s; cmp w5,#6\n\t"\
  "fmla v31.4s,v9.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N18_TAIL2(ac1, ac2) \
  "fmov v5.d[1],x10\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v16.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-112]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v17.4s,v5.4s,v"#ac2".s[0]\n\t"\
  "fmla v26.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-96]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-88]\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-80]\n\t"\
  "fmla v18.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[0]\n\t"\
  "fmla v19.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x7]\n\t"\
  "fmla v23.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "ldr d5,[x4,#-64]\n\t"\
  "fmla v29.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmla v16.4s,v4.4s,v"#ac2".s[1]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-48]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac1".s[3]; ldr x10,[x4,#-40]\n\t"\
  "fmla v26.4s,v4.4s,v"#ac2".s[3]; prfm pldl1keep,[x8]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-32]\n\t"\
  "fmla v17.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-24]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-16]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v18.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x9]\n\t"\
  "fmla v22.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "ins v8.d[1],v8.d[0]; ldr d9,[x4,#-8]\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[1]\n\t"\
  "fmla v19.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "ins v9.d[1],v9.d[0]\n\t"\
  "fmla v23.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v29.4s,v7.4s,v"#ac2".s[3]; sub w5,w5,#2\n\t"\
  "fmla v24.4s,v8.4s,v"#ac1".4s\n\t"\
  "fmla v30.4s,v8.4s,v"#ac2".4s\n\t"\
  "fmla v25.4s,v9.4s,v"#ac1".4s\n\t"\
  "fmla v31.4s,v9.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N18_FIN1 \
  "ldr s0,[x0],#4; ldr q4,[x4]; ldr q5,[x4,#16]\n\t"\
  "ldr q6,[x4,#32]; add x4,x4,#72\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v12.4s,v4.4s,v0.s[0]\n\t"\
  "fmla v13.4s,v5.4s,v0.s[0]\n\t"\
  "fmla v14.4s,v6.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v16.4s,v4.4s,v1.s[0]\n\t"\
  "fmla v17.4s,v5.4s,v1.s[0]\n\t"\
  "fmla v18.4s,v6.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4; ldr d7,[x4,#-24]\n\t"\
  "fmla v20.4s,v4.4s,v2.s[0]; ldr x10,[x4,#-16]\n\t"\
  "fmla v21.4s,v5.4s,v2.s[0]\n\t"\
  "fmla v22.4s,v6.4s,v2.s[0]\n\t"\
  "fmov v7.d[1],x10; ldr s8,[x4,#-8]\n\t"\
  "fmla v26.4s,v4.4s,v3.s[0]; ldr w10,[x4,#-4]\n\t"\
  "fmla v27.4s,v5.4s,v3.s[0]\n\t"\
  "fmla v28.4s,v6.4s,v3.s[0]\n\t"\
  "ins v8.d[1],v8.d[0]; dup v9.2d,x10\n\t"\
  "fmla v15.4s,v7.4s,v0.s[0]\n\t"\
  "fmla v19.4s,v7.4s,v1.s[0]\n\t"\
  "fmla v23.4s,v7.4s,v2.s[0]\n\t"\
  "ins v0.d[1],v2.d[0]; ins v1.d[1],v3.d[0]\n\t"\
  "fmla v29.4s,v7.4s,v3.s[0]\n\t"\
  "fmla v24.4s,v8.4s,v0.4s\n\t"\
  "fmla v30.4s,v8.4s,v1.4s\n\t"\
  "fmla v25.4s,v9.4s,v0.4s\n\t"\
  "fmla v31.4s,v9.4s,v1.4s\n\t"


/* m4n19 c_vec */
/* v10 - v13 v22_comp v23_comp v24_comp */
/* v14 - v17 v29_comp v30_comp v31_comp */
/* v18 - v21 v22_comp v23_comp v24_comp */
/* v25 - v28 v29_comp v30_comp v31_comp */

#define INIT_M4N19 \
  INIT_2V(10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N19(mode) \
  UNIT_SAVE_M4N4_VR_##mode(10, 14, 18, 25) UNIT_SAVE_M4N4_VR_##mode(11, 15, 19, 26)\
  UNIT_SAVE_M4N4_VR_##mode(12, 16, 20, 27) UNIT_SAVE_M4N4_VR_##mode(13, 17, 21, 28)\
  EDGE_SAVE_M4N1K2_##mode(22, 29) EDGE_SAVE_M4N1K2_##mode(23, 30) EDGE_SAVE_M4N1K2_##mode(24, 31)

#define KERNEL_M4N19_PRELOAD2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr x16,[x2],#8; ldr x11,[x3],#8\n\t"\
  "ldr q4,[x4]; ldr d5,[x4,#16]; ldr x10,[x4,#24]\n\t"\
  "add x4,x4,#152; fmov v0.d[1],x16\n\t"

#define KERNEL_M4N19_MAIN2(ac1, ac2, an1, an2, ap1, ap2) \
  "fmov v5.d[1],x10; ldr d"#an1",[x0],#8\n\t"\
  "fmla v10.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v14.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v18.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-120]\n\t"\
  "fmla v11.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-112]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#64]\n\t"\
  "fmla v25.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-104]\n\t"\
  "fmla v19.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-96]\n\t"\
  "fmla v26.4s,v5.4s,v"#ac2".s[2]; ldr x16,[x2],#8\n\t"\
  "fmla v12.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-88]\n\t"\
  "fmla v16.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-80]\n\t"\
  "fmla v20.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v27.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-72]\n\t"\
  "fmla v13.4s,v7.4s,v"#ac1".s[0]; ldr x10,[x4,#-64]\n\t"\
  "fmla v17.4s,v7.4s,v"#ac2".s[0]\n\t"\
  "fmla v21.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v5.d[1],x10; ldr d"#an2",[x1],#8\n\t"\
  "fmla v28.4s,v7.4s,v"#ac2".s[2]\n\t"\
  "fmla v10.4s,v4.4s,v"#ac1".s[1]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v14.4s,v4.4s,v"#ac2".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d6,[x4,#-56]\n\t"\
  "fmla v18.4s,v4.4s,v"#ac1".s[3]; ldr x10,[x4,#-48]\n\t"\
  "fmla v25.4s,v4.4s,v"#ac2".s[3]; prfm pldl1keep,[x4,#112]\n\t"\
  "fmla v11.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-40]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-32]\n\t"\
  "fmla v19.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmla v26.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-24]\n\t"\
  "fmla v12.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v16.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x4,#160]\n\t"\
  "fmla v20.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "ins v8.d[1],v8.d[0]; ldr d9,[x4,#-16]\n\t"\
  "fmla v27.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "fmla v13.4s,v7.4s,v"#ac1".s[1]; ldr x11,[x3],#8\n\t"\
  "fmla v17.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "ins v9.d[1],v9.d[0]; ldr d6,[x4,#-8]\n\t"\
  "fmla v21.4s,v7.4s,v"#ac1".s[3]; add x4,x4,#152\n\t"\
  "fmla v28.4s,v7.4s,v"#ac2".s[3]; sub w5,w5,#2\n\t"\
  "fmla v22.4s,v8.4s,v"#ac1".4s\n\t"\
  "ins v6.d[1],v6.d[0]; ldr d4,[x4,#-152]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac2".4s; ldr x10,[x4,#-144]\n\t"\
  "fmla v23.4s,v9.4s,v"#ac1".4s; cmp w5,#6\n\t"\
  "fmla v30.4s,v9.4s,v"#ac2".4s\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-136]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".4s; ldr x10,[x4,#-128]\n\t"\
  "fmla v31.4s,v6.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N19_TAIL2(ac1, ac2) \
  "fmov v5.d[1],x10\n\t"\
  "fmla v10.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v14.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v18.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-120]\n\t"\
  "fmla v11.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-112]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac2".s[0]\n\t"\
  "fmla v25.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-104]\n\t"\
  "fmla v19.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-96]\n\t"\
  "fmla v26.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v12.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-88]\n\t"\
  "fmla v16.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-80]\n\t"\
  "fmla v20.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v27.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-72]\n\t"\
  "fmla v13.4s,v7.4s,v"#ac1".s[0]; ldr x10,[x4,#-64]\n\t"\
  "fmla v17.4s,v7.4s,v"#ac2".s[0]\n\t"\
  "fmla v21.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v5.d[1],x10\n\t"\
  "fmla v28.4s,v7.4s,v"#ac2".s[2]\n\t"\
  "fmla v10.4s,v4.4s,v"#ac1".s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmla v14.4s,v4.4s,v"#ac2".s[1]\n\t"\
  "ldr d6,[x4,#-56]\n\t"\
  "fmla v18.4s,v4.4s,v"#ac1".s[3]; ldr x10,[x4,#-48]\n\t"\
  "fmla v25.4s,v4.4s,v"#ac2".s[3]; prfm pldl1keep,[x8]\n\t"\
  "fmla v11.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-40]\n\t"\
  "fmla v15.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-32]\n\t"\
  "fmla v19.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmla v26.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-24]\n\t"\
  "fmla v12.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v16.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x9]\n\t"\
  "fmla v20.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "ins v8.d[1],v8.d[0]; ldr d9,[x4,#-16]\n\t"\
  "fmla v27.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "fmla v13.4s,v7.4s,v"#ac1".s[1]\n\t"\
  "fmla v17.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "ins v9.d[1],v9.d[0]; ldr d6,[x4,#-8]\n\t"\
  "fmla v21.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v28.4s,v7.4s,v"#ac2".s[3]; sub w5,w5,#2\n\t"\
  "fmla v22.4s,v8.4s,v"#ac1".4s\n\t"\
  "ins v6.d[1],v6.d[0]\n\t"\
  "fmla v29.4s,v8.4s,v"#ac2".4s\n\t"\
  "fmla v23.4s,v9.4s,v"#ac1".4s\n\t"\
  "fmla v30.4s,v9.4s,v"#ac2".4s\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".4s\n\t"\
  "fmla v31.4s,v6.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N19_FIN1 \
  "ldr s0,[x0],#4; ldr q4,[x4]; ldr q5,[x4,#16]\n\t"\
  "ldr q6,[x4,#32]; add x4,x4,#76\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v10.4s,v4.4s,v0.s[0]\n\t"\
  "fmla v11.4s,v5.4s,v0.s[0]\n\t"\
  "fmla v12.4s,v6.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4\n\t"\
  "fmla v14.4s,v4.4s,v1.s[0]\n\t"\
  "fmla v15.4s,v5.4s,v1.s[0]\n\t"\
  "fmla v16.4s,v6.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4; ldr d7,[x4,#-28]\n\t"\
  "fmla v18.4s,v4.4s,v2.s[0]; ldr x10,[x4,#-20]\n\t"\
  "fmla v19.4s,v5.4s,v2.s[0]\n\t"\
  "fmla v20.4s,v6.4s,v2.s[0]\n\t"\
  "fmov v7.d[1],x10; ldr s8,[x4,#-12]\n\t"\
  "fmla v25.4s,v4.4s,v3.s[0]; ldr w10,[x4,#-8]\n\t"\
  "fmla v26.4s,v5.4s,v3.s[0]\n\t"\
  "fmla v27.4s,v6.4s,v3.s[0]\n\t"\
  "ins v8.d[1],v8.d[0]; dup v9.2d,x10\n\t"\
  "fmla v13.4s,v7.4s,v0.s[0]; ldr w10,[x4,#-4]\n\t"\
  "fmla v17.4s,v7.4s,v1.s[0]\n\t"\
  "fmla v21.4s,v7.4s,v2.s[0]\n\t"\
  "ins v0.d[1],v2.d[0]; ins v1.d[1],v3.d[0]\n\t"\
  "fmla v28.4s,v7.4s,v3.s[0]\n\t"\
  "fmla v22.4s,v8.4s,v0.4s\n\t"\
  "fmla v29.4s,v8.4s,v1.4s\n\t"\
  "dup v6.2d,x10\n\t"\
  "fmla v23.4s,v9.4s,v0.4s\n\t"\
  "fmla v30.4s,v9.4s,v1.4s\n\t"\
  "fmla v24.4s,v6.4s,v0.4s\n\t"\
  "fmla v31.4s,v6.4s,v1.4s\n\t"


/* m4n20 c_vec */
/* v12 - v16 */
/* v17 - v21 */
/* v22 - v26 */
/* v27 - v31 */

#define INIT_M4N20 INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N20(mode) UNIT_SAVE_M4N4_VR_##mode(12, 17, 22, 27)\
  UNIT_SAVE_M4N4_VR_##mode(13, 18, 23, 28) UNIT_SAVE_M4N4_VR_##mode(14, 19, 24, 29)\
  UNIT_SAVE_M4N4_VR_##mode(15, 20, 25, 30) UNIT_SAVE_M4N4_VR_##mode(16, 21, 26, 31)

#define KERNEL_M4N20_PRELOAD2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr x16,[x2],#8; ldr x11,[x3],#8\n\t"\
  "ldr q4,[x4]; ldr d5,[x4,#16]; ldr x10,[x4,#24]\n\t"\
  "add x4,x4,#160; fmov v0.d[1],x16\n\t"

#define KERNEL_M4N20_MAIN2(ac1, ac2, an1, an2, ap1, ap2) \
  "fmov v5.d[1],x10; ldr d"#an1",[x0],#8\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-128]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#72]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-112]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-104]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[2]; ldr x16,[x2],#8\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-96]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d4,[x4,#-80]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#120]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-64]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmov v5.d[1],x10; ldr d"#an2",[x1],#8\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".s[2]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d6,[x4,#-48]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[3]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[3]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-32]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[1]; ldr x10,[x4,#-24]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[1]; prfm pldl1keep,[x4,#168]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-16]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[1]; add x4,x4,#160\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[1]\n\t"\
  "fmov v8.d[1],x10\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[3]; sub w5,w5,#2\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[3]; ldr x11,[x3],#8\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[1]\n\t"\
  "ldr d4,[x4,#-160]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[1]; ldr x10,[x4,#-152]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[3]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]; cmp w5,#6\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "ldr d5,[x4,#-144]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".s[3]; ldr x10,[x4,#-136]\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".s[3]\n\t"

#define KERNEL_M4N20_TAIL2(ac1, ac2) \
  "fmov v5.d[1],x10\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-128]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[0]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-112]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-104]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-96]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[2]; prfm pldl1keep,[x7]\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d4,[x4,#-80]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[0]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-64]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmov v5.d[1],x10\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "ldr d6,[x4,#-48]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[3]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[3]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-32]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[1]; ldr x10,[x4,#-24]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[1]; prfm pldl1keep,[x9]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-16]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[1]\n\t"\
  "fmov v8.d[1],x10\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[3]; sub w5,w5,#2\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[1]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[3]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".s[3]\n\t"

#define KERNEL_M4N20_FIN1 \
  "ldr s0,[x0],#4; ldr q4,[x4]; ldr q5,[x4,#16]\n\t"\
  "ldr q6,[x4,#32]; add x4,x4,#80\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v12.4s,v4.4s,v0.s[0]\n\t"\
  "fmla v13.4s,v5.4s,v0.s[0]\n\t"\
  "fmla v14.4s,v6.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4; ldr d7,[x4,#-32]\n\t"\
  "fmla v17.4s,v4.4s,v1.s[0]; ldr x10,[x4,#-24]\n\t"\
  "fmla v18.4s,v5.4s,v1.s[0]\n\t"\
  "fmla v19.4s,v6.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4; fmov v7.d[1],x10\n\t"\
  "fmla v22.4s,v4.4s,v2.s[0]\n\t"\
  "fmla v23.4s,v5.4s,v2.s[0]\n\t"\
  "fmla v24.4s,v6.4s,v2.s[0]\n\t"\
  "ldr d8,[x4,#-16]\n\t"\
  "fmla v27.4s,v4.4s,v3.s[0]; ldr x10,[x4,#-8]\n\t"\
  "fmla v28.4s,v5.4s,v3.s[0]\n\t"\
  "fmla v29.4s,v6.4s,v3.s[0]\n\t"\
  "fmov v8.d[1],x10\n\t"\
  "fmla v15.4s,v7.4s,v0.s[0]\n\t"\
  "fmla v20.4s,v7.4s,v1.s[0]\n\t"\
  "fmla v25.4s,v7.4s,v2.s[0]\n\t"\
  "fmla v30.4s,v7.4s,v3.s[0]\n\t"\
  "fmla v16.4s,v8.4s,v0.s[0]\n\t"\
  "fmla v21.4s,v8.4s,v1.s[0]\n\t"\
  "fmla v26.4s,v8.4s,v2.s[0]\n\t"\
  "fmla v31.4s,v8.4s,v3.s[0]\n\t"


/* m4n21 c_vec */
/* v12 - v16 v10_comp */
/* v17 - v21 v11_comp */
/* v22 - v26 v10_comp */
/* v27 - v31 v11_comp */

#define INIT_M4N21 \
  INIT_2V(10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N21(mode) UNIT_SAVE_M4N4_VR_##mode(12, 17, 22, 27)\
  UNIT_SAVE_M4N4_VR_##mode(13, 18, 23, 28) UNIT_SAVE_M4N4_VR_##mode(14, 19, 24, 29)\
  UNIT_SAVE_M4N4_VR_##mode(15, 20, 25, 30) UNIT_SAVE_M4N4_VR_##mode(16, 21, 26, 31)\
  EDGE_SAVE_M4N1K2_##mode(10, 11)

#define KERNEL_M4N21_PRELOAD2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr x16,[x2],#8; ldr x11,[x3],#8\n\t"\
  "ldr q4,[x4]; ldr d5,[x4,#16]; ldr x10,[x4,#24]\n\t"\
  "add x4,x4,#168; fmov v0.d[1],x16\n\t"

#define KERNEL_M4N21_MAIN2(ac1, ac2, an1, an2, ap1, ap2) \
  "fmov v5.d[1],x10; ldr d"#an1",[x0],#8\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-136]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-128]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#64]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-120]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-112]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-104]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-96]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[2]; ldr x16,[x2],#8\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d4,[x4,#-88]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[0]; ldr x10,[x4,#-80]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#120]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-72]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-64]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmov v5.d[1],x10; ldr d"#an2",[x1],#8\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".s[2]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d6,[x4,#-56]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[1]; ldr x10,[x4,#-48]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[3]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[3]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-40]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[1]; ldr x10,[x4,#-32]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[1]; prfm pldl1keep,[x4,#176]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-24]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[3]; ldr x10,[x4,#-16]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-8]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[3]; add x4,x4,#168\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[3]; ldr x11,[x3],#8\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[1]\n\t"\
  "ins v9.d[1],v9.d[0]; ldr d4,[x4,#-168]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[1]; ldr x10,[x4,#-160]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[3]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]; sub w5,w5,#2\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "ldr d5,[x4,#-152]\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".s[3]; ldr x10,[x4,#-144]\n\t"\
  "fmla v10.4s,v9.4s,v"#ac1".4s; cmp w5,#6\n\t"\
  "fmla v11.4s,v9.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N21_TAIL2(ac1, ac2) \
  "fmov v5.d[1],x10\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[0]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-136]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-128]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-120]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-112]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-104]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-96]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v8.d[1],x10; ldr d4,[x4,#-88]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[0]; ldr x10,[x4,#-80]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x7]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-72]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-64]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[0]\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[0]\n\t"\
  "fmov v5.d[1],x10\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".s[2]\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "ldr d6,[x4,#-56]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[1]; ldr x10,[x4,#-48]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[3]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[3]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-40]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[1]; ldr x10,[x4,#-32]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[1]; prfm pldl1keep,[x9]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d8,[x4,#-24]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[3]; ldr x10,[x4,#-16]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[1]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[1]\n\t"\
  "fmov v8.d[1],x10; ldr d9,[x4,#-8]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[3]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[1]\n\t"\
  "ins v9.d[1],v9.d[0]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[3]\n\t"\
  "fmla v16.4s,v8.4s,v"#ac1".s[1]; sub w5,w5,#2\n\t"\
  "fmla v21.4s,v8.4s,v"#ac2".s[1]\n\t"\
  "fmla v26.4s,v8.4s,v"#ac1".s[3]\n\t"\
  "fmla v31.4s,v8.4s,v"#ac2".s[3]\n\t"\
  "fmla v10.4s,v9.4s,v"#ac1".4s\n\t"\
  "fmla v11.4s,v9.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N21_FIN1 \
  "ldr s0,[x0],#4; ldr q4,[x4]; ldr q5,[x4,#16]\n\t"\
  "ldr q6,[x4,#32]; add x4,x4,#84\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v12.4s,v4.4s,v0.s[0]\n\t"\
  "fmla v13.4s,v5.4s,v0.s[0]\n\t"\
  "fmla v14.4s,v6.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4; ldr d7,[x4,#-36]\n\t"\
  "fmla v17.4s,v4.4s,v1.s[0]; ldr x10,[x4,#-28]\n\t"\
  "fmla v18.4s,v5.4s,v1.s[0]\n\t"\
  "fmla v19.4s,v6.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4; fmov v7.d[1],x10\n\t"\
  "fmla v22.4s,v4.4s,v2.s[0]\n\t"\
  "fmla v23.4s,v5.4s,v2.s[0]\n\t"\
  "fmla v24.4s,v6.4s,v2.s[0]\n\t"\
  "ldr d8,[x4,#-20]\n\t"\
  "fmla v27.4s,v4.4s,v3.s[0]; ldr x10,[x4,#-12]\n\t"\
  "fmla v28.4s,v5.4s,v3.s[0]\n\t"\
  "fmla v29.4s,v6.4s,v3.s[0]\n\t"\
  "fmov v8.d[1],x10; ldr s9,[x4,#-4]\n\t"\
  "fmla v15.4s,v7.4s,v0.s[0]\n\t"\
  "fmla v20.4s,v7.4s,v1.s[0]\n\t"\
  "fmla v25.4s,v7.4s,v2.s[0]\n\t"\
  "ins v9.d[1],v9.d[0]\n\t"\
  "fmla v30.4s,v7.4s,v3.s[0]\n\t"\
  "fmla v16.4s,v8.4s,v0.s[0]\n\t"\
  "fmla v21.4s,v8.4s,v1.s[0]\n\t"\
  "ins v0.d[1],v2.d[0]; ins v1.d[1],v3.d[0]\n\t"\
  "fmla v26.4s,v8.4s,v2.s[0]\n\t"\
  "fmla v31.4s,v8.4s,v3.s[0]\n\t"\
  "fmla v10.4s,v9.4s,v0.4s\n\t"\
  "fmla v11.4s,v9.4s,v1.4s\n\t"


/* m4n22 c_vec */
/* v12 - v16 v10_comp v8_comp */
/* v17 - v21 v11_comp v9_comp */
/* v22 - v26 v10_comp v8_comp */
/* v27 - v31 v11_comp v9_comp */

#define INIT_M4N22 \
  INIT_4V(8, 9, 10, 11) INIT_4V(12, 13, 14, 15)\
  INIT_4V(16, 17, 18, 19) INIT_4V(20, 21, 22, 23)\
  INIT_4V(24, 25, 26, 27) INIT_4V(28, 29, 30, 31)

#define SAVE_M4N22(mode) UNIT_SAVE_M4N4_VR_##mode(12, 17, 22, 27)\
  UNIT_SAVE_M4N4_VR_##mode(13, 18, 23, 28) UNIT_SAVE_M4N4_VR_##mode(14, 19, 24, 29)\
  UNIT_SAVE_M4N4_VR_##mode(15, 20, 25, 30) UNIT_SAVE_M4N4_VR_##mode(16, 21, 26, 31)\
  EDGE_SAVE_M4N1K2_##mode(10, 11) EDGE_SAVE_M4N1K2_##mode(8, 9)

#define KERNEL_M4N22_PRELOAD2 \
  "ldr d0,[x0],#8; ldr d1,[x1],#8; ldr x16,[x2],#8; ldr x11,[x3],#8\n\t"\
  "ldr q4,[x4]; ldr d5,[x4,#16]; ldr x10,[x4,#24]\n\t"\
  "add x4,x4,#176; fmov v0.d[1],x16\n\t"

#define KERNEL_M4N22_MAIN2(ac1, ac2, an1, an2, ap1, ap2) \
  "fmov v5.d[1],x10; ldr d"#an1",[x0],#8\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-144]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-136]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#64]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-128]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-120]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-112]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[2]; ldr x16,[x2],#8\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-96]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x4,#120]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-80]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v16.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v21.4s,v4.4s,v"#ac2".s[0]\n\t"\
  "fmov v6.d[1],x10; ldr d"#an2",[x1],#8\n\t"\
  "fmla v26.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmla v31.4s,v4.4s,v"#ac2".s[2]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v12.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmov v"#an1".d[1],x16; ldr d7,[x4,#-64]\n\t"\
  "fmla v17.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-56]\n\t"\
  "fmla v22.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-48]\n\t"\
  "fmla v13.4s,v6.4s,v"#ac1".s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v18.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x4,#176]\n\t"\
  "fmla v23.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-32]\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v14.4s,v7.4s,v"#ac1".s[1]\n\t"\
  "fmla v19.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-16]\n\t"\
  "fmla v24.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v29.4s,v7.4s,v"#ac2".s[3]; ldr x11,[x3],#8\n\t"\
  "fmla v15.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "ins v6.d[1],v6.d[0]; ldr d7,[x4,#-8]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac2".s[1]; add x4,x4,#176\n\t"\
  "fmla v25.4s,v4.4s,v"#ac1".s[3]\n\t"\
  "fmla v30.4s,v4.4s,v"#ac2".s[3]\n\t"\
  "ins v7.d[1],v7.d[0]; ldr d4,[x4,#-176]\n\t"\
  "fmla v16.4s,v5.4s,v"#ac1".s[1]; ldr x10,[x4,#-168]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac2".s[1]; sub w5,w5,#2\n\t"\
  "fmla v26.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v31.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmla v10.4s,v6.4s,v"#ac1".4s; cmp w5,#6\n\t"\
  "fmla v11.4s,v6.4s,v"#ac2".4s\n\t"\
  "ldr d5,[x4,#-160]\n\t"\
  "fmla v8.4s,v7.4s,v"#ac1".4s; ldr x10,[x4,#-152]\n\t"\
  "fmla v9.4s,v7.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N22_TAIL2(ac1, ac2) \
  "fmov v5.d[1],x10\n\t"\
  "fmla v12.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v17.4s,v4.4s,v"#ac2".s[0]; prfm pldl1keep,[x6]\n\t"\
  "fmla v22.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmov v"#ac2".d[1],x11; ldr d6,[x4,#-144]\n\t"\
  "fmla v13.4s,v5.4s,v"#ac1".s[0]; ldr x10,[x4,#-136]\n\t"\
  "fmla v18.4s,v5.4s,v"#ac2".s[0]\n\t"\
  "fmla v27.4s,v4.4s,v"#ac2".s[2]\n\t"\
  "fmov v6.d[1],x10; ldr d7,[x4,#-128]\n\t"\
  "fmla v23.4s,v5.4s,v"#ac1".s[2]; ldr x10,[x4,#-120]\n\t"\
  "fmla v28.4s,v5.4s,v"#ac2".s[2]\n\t"\
  "fmla v14.4s,v6.4s,v"#ac1".s[0]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-112]\n\t"\
  "fmla v19.4s,v6.4s,v"#ac2".s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v24.4s,v6.4s,v"#ac1".s[2]\n\t"\
  "fmla v29.4s,v6.4s,v"#ac2".s[2]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-96]\n\t"\
  "fmla v15.4s,v7.4s,v"#ac1".s[0]; ldr x10,[x4,#-88]\n\t"\
  "fmla v20.4s,v7.4s,v"#ac2".s[0]; prfm pldl1keep,[x7]\n\t"\
  "fmla v25.4s,v7.4s,v"#ac1".s[2]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-80]\n\t"\
  "fmla v30.4s,v7.4s,v"#ac2".s[2]; ldr x10,[x4,#-72]\n\t"\
  "fmla v16.4s,v4.4s,v"#ac1".s[0]\n\t"\
  "fmla v21.4s,v4.4s,v"#ac2".s[0]\n\t"\
  "fmov v6.d[1],x10\n\t"\
  "fmla v26.4s,v4.4s,v"#ac1".s[2]\n\t"\
  "fmla v31.4s,v4.4s,v"#ac2".s[2]; prfm pldl1keep,[x8]\n\t"\
  "fmla v12.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "ldr d7,[x4,#-64]\n\t"\
  "fmla v17.4s,v5.4s,v"#ac2".s[1]; ldr x10,[x4,#-56]\n\t"\
  "fmla v22.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmla v27.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmov v7.d[1],x10; ldr d4,[x4,#-48]\n\t"\
  "fmla v13.4s,v6.4s,v"#ac1".s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v18.4s,v6.4s,v"#ac2".s[1]; prfm pldl1keep,[x9]\n\t"\
  "fmla v23.4s,v6.4s,v"#ac1".s[3]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-32]\n\t"\
  "fmla v28.4s,v6.4s,v"#ac2".s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v14.4s,v7.4s,v"#ac1".s[1]\n\t"\
  "fmla v19.4s,v7.4s,v"#ac2".s[1]\n\t"\
  "fmov v5.d[1],x10; ldr d6,[x4,#-16]\n\t"\
  "fmla v24.4s,v7.4s,v"#ac1".s[3]\n\t"\
  "fmla v29.4s,v7.4s,v"#ac2".s[3]\n\t"\
  "fmla v15.4s,v4.4s,v"#ac1".s[1]\n\t"\
  "ins v6.d[1],v6.d[0]; ldr d7,[x4,#-8]\n\t"\
  "fmla v20.4s,v4.4s,v"#ac2".s[1]\n\t"\
  "fmla v25.4s,v4.4s,v"#ac1".s[3]\n\t"\
  "fmla v30.4s,v4.4s,v"#ac2".s[3]\n\t"\
  "ins v7.d[1],v7.d[0]\n\t"\
  "fmla v16.4s,v5.4s,v"#ac1".s[1]\n\t"\
  "fmla v21.4s,v5.4s,v"#ac2".s[1]; sub w5,w5,#2\n\t"\
  "fmla v26.4s,v5.4s,v"#ac1".s[3]\n\t"\
  "fmla v31.4s,v5.4s,v"#ac2".s[3]\n\t"\
  "fmla v10.4s,v6.4s,v"#ac1".4s\n\t"\
  "fmla v11.4s,v6.4s,v"#ac2".4s\n\t"\
  "fmla v8.4s,v7.4s,v"#ac1".4s\n\t"\
  "fmla v9.4s,v7.4s,v"#ac2".4s\n\t"

#define KERNEL_M4N22_FIN1 \
  "ldr s0,[x0],#4; ldr q4,[x4]; ldr q5,[x4,#16]\n\t"\
  "ldr q6,[x4,#32]; add x4,x4,#88\n\t"\
  "ldr s1,[x1],#4\n\t"\
  "fmla v12.4s,v4.4s,v0.s[0]\n\t"\
  "fmla v13.4s,v5.4s,v0.s[0]\n\t"\
  "fmla v14.4s,v6.4s,v0.s[0]\n\t"\
  "ldr s2,[x2],#4; ldr d7,[x4,#-40]\n\t"\
  "fmla v17.4s,v4.4s,v1.s[0]; ldr x10,[x4,#-32]\n\t"\
  "fmla v18.4s,v5.4s,v1.s[0]\n\t"\
  "fmla v19.4s,v6.4s,v1.s[0]\n\t"\
  "ldr s3,[x3],#4; fmov v7.d[1],x10\n\t"\
  "fmla v22.4s,v4.4s,v2.s[0]\n\t"\
  "fmla v23.4s,v5.4s,v2.s[0]\n\t"\
  "fmla v27.4s,v4.4s,v3.s[0]\n\t"\
  "ldr d4,[x4,#-24]\n\t"\
  "fmla v24.4s,v6.4s,v2.s[0]; ldr x10,[x4,#-16]\n\t"\
  "fmla v28.4s,v5.4s,v3.s[0]\n\t"\
  "fmla v29.4s,v6.4s,v3.s[0]\n\t"\
  "fmov v4.d[1],x10; ldr s5,[x4,#-8]\n\t"\
  "fmla v15.4s,v7.4s,v0.s[0]; ldr w11,[x4,#-4]\n\t"\
  "fmla v20.4s,v7.4s,v1.s[0]\n\t"\
  "fmla v25.4s,v7.4s,v2.s[0]\n\t"\
  "ins v5.d[1],v5.d[0]; dup v6.2d,x11\n\t"\
  "fmla v30.4s,v7.4s,v3.s[0]\n\t"\
  "fmla v16.4s,v4.4s,v0.s[0]\n\t"\
  "fmla v21.4s,v4.4s,v1.s[0]\n\t"\
  "ins v0.d[1],v2.d[0]; ins v1.d[1],v3.d[0]\n\t"\
  "fmla v26.4s,v4.4s,v2.s[0]\n\t"\
  "fmla v31.4s,v4.4s,v3.s[0]\n\t"\
  "fmla v10.4s,v5.4s,v0.4s\n\t"\
  "fmla v11.4s,v5.4s,v1.4s\n\t"\
  "fmla v8.4s,v6.4s,v0.4s\n\t"\
  "fmla v9.4s,v6.4s,v1.4s\n\t"


#define FUNC_K2(ndim) \
static inline void sgemm_skinny1_a53_m4n##ndim(\
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
    "cmp w5,#2; b.lt 4f\n\t"\
    KERNEL_M4N##ndim##_PRELOAD2\
    "cmp w5,#6; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    KERNEL_M4N##ndim##_MAIN2(0, 1, 2, 3, 0, 1)\
    KERNEL_M4N##ndim##_MAIN2(2, 3, 0, 1, 2, 3)\
    "b.ge 1b; 2:\n\t"\
    "cmp w5,#4; b.lt 3f\n\t"\
    KERNEL_M4N##ndim##_MAIN2(0, 1, 2, 3, 0, 1)\
    KERNEL_M4N##ndim##_TAIL2(2, 3)\
    "b 4f; 3:\n\t"\
    KERNEL_M4N##ndim##_TAIL2(0, 1)\
    "4:\n\t"\
    "cmp w5,#1; b.lt 6f\n\t"\
    "5:\n\t"\
    KERNEL_M4N##ndim##_FIN1\
    "6:\n\t"\
    INIT_SAVE\
    "cmp %w[c_rowmajor],#0; b.eq 7f\n\t"\
    SAVE_M4N##ndim(CR) "b 8f\n\t"\
    "7:\n\t"\
    SAVE_M4N##ndim(CC)\
    "8:\n\t"\
  ::[a_ptr]"r"(a_ptr), [c_ptr]"r"(c_ptr), [b_scr]"r"(b_scr),\
    [K]"r"(K), [LDA]"r"(LDA), [LDC]"r"(LDC),\
    [beta_addr]"r"(beta_addr), [c_rowmajor]"r"(c_rowmajor)\
  :"cc","memory","x0","x1","x2","x3","x4","x5","x6","x7","x8","x9",\
  "x10","x11","x12","x13","x14","x15","x16",\
  "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13",\
  "v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25",\
  "v26","v27","v28","v29","v30","v31");\
}

FUNC_K2(15)
FUNC_K2(16)
FUNC_K2(17)
FUNC_K2(18)
FUNC_K2(19)
FUNC_K2(20)
FUNC_K2(21)
FUNC_K2(22)

#define INIT_M4N23 INIT_4V(6, 7, 8, 9) \
  INIT_4V(10, 11, 12, 13) INIT_4V(14, 15, 16, 17)\
  INIT_4V(18, 19, 20, 21) INIT_4V(22, 23, 24, 25)\
  INIT_2V(26, 27) INIT_1V(28)

#define SAVE_M4N23(mode) \
  UNIT_SAVE_M4N4_VC_##mode(6, 7, 8, 9) UNIT_SAVE_M4N4_VC_##mode(10, 11, 12, 13)\
  UNIT_SAVE_M4N4_VC_##mode(14, 15, 16, 17) UNIT_SAVE_M4N4_VC_##mode(18, 19, 20, 21)\
  UNIT_SAVE_M4N4_VC_##mode(22, 23, 24, 25) EDGE_SAVE_M4N1K1_##mode(26)\
  EDGE_SAVE_M4N1K1_##mode(27) EDGE_SAVE_M4N1K1_##mode(28)

#define KERNEL_M4N23_PRELOAD2 \
  "ldr x16,[x0],#8; ldr x17,[x1],#8; ldr x19,[x2],#8; ldr x20,[x3],#8\n\t"\
  "ldr q2,[x4]; ldr q3,[x4,#16]; ldr x10,[x4,#24]; add x4,x4,#184\n\t"\
  "mov w11,w16; bfi x11,x17,#32,#32; fmov d0,x11\n\t"\
  "mov w11,w19; bfi x11,x20,#32,#32; fmov v0.d[1],x11\n\t"

#define KERNEL_M4N23_MAIN2(ap1, ap2) \
  "fmov v3.d[1],x10; ldr d4,[x4,#-152]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-144]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]; bfxil x17,x16,#32,#32\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d1,x17\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]; ldr x16,[x0],#8\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]; bfxil x20,x19,#32,#32\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "fmov v1.d[1],x20; ldr d2,[x4,#-136]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-128]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x4,#48]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-120]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-112]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]; ldr x17,[x1],#8\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-104]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-96]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]; mov w11,w16\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]; bfi x11,x17,#32,#32\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]; ldr x19,[x2],#8\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-88]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-80]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-72]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-64]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v6.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-56]\n\t"\
  "fmla v7.4s,v1.4s,v2.s[0]; ldr x10,[x4,#-48]\n\t"\
  "fmla v8.4s,v1.4s,v2.s[1]; prfm pldl1keep,[x4,#112]\n\t"\
  "fmla v9.4s,v1.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d0,x11\n\t"\
  "fmla v10.4s,v1.4s,v2.s[3]; mov w11,w19\n\t"\
  "fmla v11.4s,v1.4s,v3.s[0]; ldr x20,[x3],#8\n\t"\
  "fmla v12.4s,v1.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-40]\n\t"\
  "fmla v13.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-32]\n\t"\
  "fmla v14.4s,v1.4s,v3.s[3]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v15.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-24]\n\t"\
  "fmla v16.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-16]\n\t"\
  "fmla v17.4s,v1.4s,v4.s[2]; bfi x11,x20,#32,#32\n\t"\
  "fmla v18.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; fmov v0.d[1],x11\n\t"\
  "fmla v19.4s,v1.4s,v2.s[0]; sub w5,w5,#2\n\t"\
  "fmla v20.4s,v1.4s,v2.s[1]; cmp w5,#6\n\t"\
  "fmla v21.4s,v1.4s,v2.s[2]\n\t"\
  "ldr d4,[x4,#-8]\n\t"\
  "fmla v22.4s,v1.4s,v2.s[3]; prfm pldl1keep,[x4,#176]\n\t"\
  "fmla v23.4s,v1.4s,v3.s[0]; add x4,x4,#184\n\t"\
  "fmla v24.4s,v1.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-184]\n\t"\
  "fmla v25.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-176]\n\t"\
  "fmla v26.4s,v1.4s,v3.s[3]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-168]\n\t"\
  "fmla v27.4s,v1.4s,v4.s[0]; ldr x10,[x4,#-160]\n\t"\
  "fmla v28.4s,v1.4s,v4.s[1]\n\t"

#define KERNEL_M4N23_TAIL2 \
  "fmov v3.d[1],x10; ldr d4,[x4,#-152]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-144]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]; bfxil x17,x16,#32,#32\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d1,x17\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]; bfxil x20,x19,#32,#32\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "fmov v1.d[1],x20; ldr d2,[x4,#-136]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-128]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-120]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-112]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-104]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-96]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-88]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-80]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x6]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-72]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-64]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v6.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10\n\t"\
  "fmla v7.4s,v1.4s,v2.s[0]\n\t"\
  "fmla v8.4s,v1.4s,v2.s[1]\n\t"\
  "fmla v9.4s,v1.4s,v2.s[2]\n\t"\
  "ldr d4,[x4,#-56]\n\t"\
  "fmla v10.4s,v1.4s,v2.s[3]; ldr x10,[x4,#-48]\n\t"\
  "fmla v11.4s,v1.4s,v3.s[0]; prfm pldl1keep,[x7]\n\t"\
  "fmla v12.4s,v1.4s,v3.s[1]\n\t"\
  "fmov v4.d[1],x10; ldr d2,[x4,#-40]\n\t"\
  "fmla v13.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-32]\n\t"\
  "fmla v14.4s,v1.4s,v3.s[3]; prfm pldl1keep,[x8]\n\t"\
  "fmla v15.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-24]\n\t"\
  "fmla v16.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-16]\n\t"\
  "fmla v17.4s,v1.4s,v4.s[2]; prfm pldl1keep,[x9]\n\t"\
  "fmla v18.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10\n\t"\
  "fmla v19.4s,v1.4s,v2.s[0]; sub w5,w5,#2\n\t"\
  "fmla v20.4s,v1.4s,v2.s[1]\n\t"\
  "fmla v21.4s,v1.4s,v2.s[2]\n\t"\
  "ldr d4,[x4,#-8]\n\t"\
  "fmla v22.4s,v1.4s,v2.s[3]\n\t"\
  "fmla v23.4s,v1.4s,v3.s[0]\n\t"\
  "fmla v24.4s,v1.4s,v3.s[1]\n\t"\
  "fmla v25.4s,v1.4s,v3.s[2]\n\t"\
  "fmla v26.4s,v1.4s,v3.s[3]\n\t"\
  "fmla v27.4s,v1.4s,v4.s[0]\n\t"\
  "fmla v28.4s,v1.4s,v4.s[1]\n\t"

#define KERNEL_M4N23_FIN1 \
  "ldr w16,[x0],#4; ldr q2,[x4]\n\t"\
  "ldr w17,[x1],#4; ldr d3,[x4,#16]\n\t"\
  "ldr w19,[x2],#4; ldr x10,[x4,#24]\n\t"\
  "ldr w20,[x3],#4; orr x16,x16,x17,LSL #32\n\t"\
  "fmov d0,x16; orr x19,x19,x20,LSL #32; fmov v0.d[1],x19\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#32]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#40]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#48]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#56]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#64]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#72]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#80]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr w10,[x4,#88]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]; add x4,x4,#92\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"


#define INIT_M4N24 INIT_4V(6, 7, 8, 9) \
  INIT_4V(10, 11, 12, 13) INIT_4V(14, 15, 16, 17)\
  INIT_4V(18, 19, 20, 21) INIT_4V(22, 23, 24, 25)\
  INIT_4V(26, 27, 28, 29)

#define SAVE_M4N24(mode) \
  UNIT_SAVE_M4N4_VC_##mode(6, 7, 8, 9) UNIT_SAVE_M4N4_VC_##mode(10, 11, 12, 13)\
  UNIT_SAVE_M4N4_VC_##mode(14, 15, 16, 17) UNIT_SAVE_M4N4_VC_##mode(18, 19, 20, 21)\
  UNIT_SAVE_M4N4_VC_##mode(22, 23, 24, 25) UNIT_SAVE_M4N4_VC_##mode(26, 27, 28, 29)

#define KERNEL_M4N24_PRELOAD2 \
  "ldr x16,[x0],#8; ldr x17,[x1],#8; ldr x19,[x2],#8; ldr x20,[x3],#8\n\t"\
  "ldr q2,[x4]; ldr q3,[x4,#16]; ldr x10,[x4,#24]; add x4,x4,#192\n\t"\
  "mov w11,w16; bfi x11,x17,#32,#32; fmov d0,x11\n\t"\
  "mov w11,w19; bfi x11,x20,#32,#32; fmov v0.d[1],x11\n\t"

#define KERNEL_M4N24_MAIN2(ap1, ap2) \
  "fmov v3.d[1],x10; ldr d4,[x4,#-160]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-152]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]; bfxil x17,x16,#32,#32\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d1,x17\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]; ldr x16,[x0],#8\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]; bfxil x20,x19,#32,#32\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "fmov v1.d[1],x20; ldr d2,[x4,#-144]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-136]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x4,#48]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-128]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-120]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]; ldr x17,[x1],#8\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-112]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]; mov w11,w16\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]; bfi x11,x17,#32,#32\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]; ldr x19,[x2],#8\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-96]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-88]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-80]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-72]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v29.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-64]\n\t"\
  "fmla v6.4s,v1.4s,v2.s[0]; ldr x10,[x4,#-56]\n\t"\
  "fmla v7.4s,v1.4s,v2.s[1]; prfm pldl1keep,[x4,#112]\n\t"\
  "fmla v8.4s,v1.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d0,x11\n\t"\
  "fmla v9.4s,v1.4s,v2.s[3]; mov w11,w19\n\t"\
  "fmla v10.4s,v1.4s,v3.s[0]; ldr x20,[x3],#8\n\t"\
  "fmla v11.4s,v1.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-48]\n\t"\
  "fmla v12.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v13.4s,v1.4s,v3.s[3]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v14.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-32]\n\t"\
  "fmla v15.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-24]\n\t"\
  "fmla v16.4s,v1.4s,v4.s[2]; bfi x11,x20,#32,#32\n\t"\
  "fmla v17.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; fmov v0.d[1],x11\n\t"\
  "fmla v18.4s,v1.4s,v2.s[0]; sub w5,w5,#2\n\t"\
  "fmla v19.4s,v1.4s,v2.s[1]; cmp w5,#6\n\t"\
  "fmla v20.4s,v1.4s,v2.s[2]\n\t"\
  "ldr d4,[x4,#-16]\n\t"\
  "fmla v21.4s,v1.4s,v2.s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v22.4s,v1.4s,v3.s[0]; prfm pldl1keep,[x4,#176]\n\t"\
  "fmla v23.4s,v1.4s,v3.s[1]\n\t"\
  "fmov v4.d[1],x10; ldr d2,[x4]\n\t"\
  "fmla v24.4s,v1.4s,v3.s[2]; ldr x10,[x4,#8]\n\t"\
  "fmla v25.4s,v1.4s,v3.s[3]; add x4,x4,#192\n\t"\
  "fmla v26.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-176]\n\t"\
  "fmla v27.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-168]\n\t"\
  "fmla v28.4s,v1.4s,v4.s[2]\n\t"\
  "fmla v29.4s,v1.4s,v4.s[3]\n\t"

#define KERNEL_M4N24_TAIL2 \
  "fmov v3.d[1],x10; ldr d4,[x4,#-160]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-152]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]; bfxil x17,x16,#32,#32\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d1,x17\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]; bfxil x20,x19,#32,#32\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "fmov v1.d[1],x20; ldr d2,[x4,#-144]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-136]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-128]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-120]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-112]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-104]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-96]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-88]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x6]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-80]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-72]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v29.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-64]\n\t"\
  "fmla v6.4s,v1.4s,v2.s[0]; ldr x10,[x4,#-56]\n\t"\
  "fmla v7.4s,v1.4s,v2.s[1]; prfm pldl1keep,[x7]\n\t"\
  "fmla v8.4s,v1.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v9.4s,v1.4s,v2.s[3]\n\t"\
  "fmla v10.4s,v1.4s,v3.s[0]\n\t"\
  "fmla v11.4s,v1.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-48]\n\t"\
  "fmla v12.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-40]\n\t"\
  "fmla v13.4s,v1.4s,v3.s[3]; prfm pldl1keep,[x8]\n\t"\
  "fmla v14.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-32]\n\t"\
  "fmla v15.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-24]\n\t"\
  "fmla v16.4s,v1.4s,v4.s[2]\n\t"\
  "fmla v17.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10\n\t"\
  "fmla v18.4s,v1.4s,v2.s[0]; sub w5,w5,#2\n\t"\
  "fmla v19.4s,v1.4s,v2.s[1]\n\t"\
  "fmla v20.4s,v1.4s,v2.s[2]\n\t"\
  "ldr d4,[x4,#-16]\n\t"\
  "fmla v21.4s,v1.4s,v2.s[3]; ldr x10,[x4,#-8]\n\t"\
  "fmla v22.4s,v1.4s,v3.s[0]; prfm pldl1keep,[x9]\n\t"\
  "fmla v23.4s,v1.4s,v3.s[1]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v24.4s,v1.4s,v3.s[2]\n\t"\
  "fmla v25.4s,v1.4s,v3.s[3]\n\t"\
  "fmla v26.4s,v1.4s,v4.s[0]\n\t"\
  "fmla v27.4s,v1.4s,v4.s[1]\n\t"\
  "fmla v28.4s,v1.4s,v4.s[2]\n\t"\
  "fmla v29.4s,v1.4s,v4.s[3]\n\t"

#define KERNEL_M4N24_FIN1 \
  "ldr w16,[x0],#4; ldr q2,[x4]\n\t"\
  "ldr w17,[x1],#4; ldr d3,[x4,#16]\n\t"\
  "ldr w19,[x2],#4; ldr x10,[x4,#24]\n\t"\
  "ldr w20,[x3],#4; orr x16,x16,x17,LSL #32\n\t"\
  "fmov d0,x16; orr x19,x19,x20,LSL #32; fmov v0.d[1],x19\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#32]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#40]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#48]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#56]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#64]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#72]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#80]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#88]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]; add x4,x4,#96\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v29.4s,v0.4s,v4.s[3]\n\t"


#define INIT_M4N25 INIT_4V(6, 7, 8, 9) \
  INIT_4V(10, 11, 12, 13) INIT_4V(14, 15, 16, 17)\
  INIT_4V(18, 19, 20, 21) INIT_4V(22, 23, 24, 25)\
  INIT_4V(26, 27, 28, 29) INIT_1V(30)

#define SAVE_M4N25(mode) \
  UNIT_SAVE_M4N4_VC_##mode(6, 7, 8, 9) UNIT_SAVE_M4N4_VC_##mode(10, 11, 12, 13)\
  UNIT_SAVE_M4N4_VC_##mode(14, 15, 16, 17) UNIT_SAVE_M4N4_VC_##mode(18, 19, 20, 21)\
  UNIT_SAVE_M4N4_VC_##mode(22, 23, 24, 25) UNIT_SAVE_M4N4_VC_##mode(26, 27, 28, 29)\
  EDGE_SAVE_M4N1K1_##mode(30)

#define KERNEL_M4N25_PRELOAD2 \
  "ldr x16,[x0],#8; ldr x17,[x1],#8; ldr x19,[x2],#8; ldr x20,[x3],#8\n\t"\
  "ldr q2,[x4]; ldr q3,[x4,#16]; ldr x10,[x4,#24]; add x4,x4,#200\n\t"\
  "mov w11,w16; bfi x11,x17,#32,#32; fmov d0,x11\n\t"\
  "mov w11,w19; bfi x11,x20,#32,#32; fmov v0.d[1],x11\n\t"

#define KERNEL_M4N25_MAIN2(ap1, ap2) \
  "fmov v3.d[1],x10; ldr d4,[x4,#-168]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-160]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]; bfxil x17,x16,#32,#32\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d1,x17\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]; ldr x16,[x0],#8\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]; bfxil x20,x19,#32,#32\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "fmov v1.d[1],x20; ldr d2,[x4,#-152]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-144]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x4,#48]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-136]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-128]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]; ldr x17,[x1],#8\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-120]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-112]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]; mov w11,w16\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]; bfi x11,x17,#32,#32\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-104]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-96]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-88]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-80]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]; ldr x19,[x2],#8\n\t"\
  "fmla v29.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-72]\n\t"\
  "fmla v30.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-64]\n\t"\
  "fmla v6.4s,v1.4s,v2.s[1]; prfm pldl1keep,[x4,#96]\n\t"\
  "fmla v7.4s,v1.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d0,x11\n\t"\
  "fmla v8.4s,v1.4s,v2.s[3]; mov w11,w19\n\t"\
  "fmla v9.4s,v1.4s,v3.s[0]\n\t"\
  "fmla v10.4s,v1.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-56]\n\t"\
  "fmla v11.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-48]\n\t"\
  "fmla v12.4s,v1.4s,v3.s[3]; ldr x20,[x3],#8\n\t"\
  "fmla v13.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-40]\n\t"\
  "fmla v14.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-32]\n\t"\
  "fmla v15.4s,v1.4s,v4.s[2]; sub w5,w5,#2\n\t"\
  "fmla v16.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10\n\t"\
  "fmla v17.4s,v1.4s,v2.s[0]; bfi x11,x20,#32,#32\n\t"\
  "fmla v18.4s,v1.4s,v2.s[1]; cmp w5,#6\n\t"\
  "fmla v19.4s,v1.4s,v2.s[2]\n\t"\
  "ldr d4,[x4,#-24]; fmov v0.d[1],x11\n\t"\
  "fmla v20.4s,v1.4s,v2.s[3]; ldr x10,[x4,#-16]\n\t"\
  "fmla v21.4s,v1.4s,v3.s[0]; prfm pldl1keep,[x4,#144]\n\t"\
  "fmla v22.4s,v1.4s,v3.s[1]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-8]\n\t"\
  "fmla v23.4s,v1.4s,v3.s[2]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v24.4s,v1.4s,v3.s[3]; add x4,x4,#200\n\t"\
  "fmla v25.4s,v1.4s,v4.s[0]\n\t"\
  "ldr d2,[x4,#-200]\n\t"\
  "fmla v26.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-192]\n\t"\
  "fmla v27.4s,v1.4s,v4.s[2]; prfm pldl1keep,[x4]\n\t"\
  "fmla v28.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-184]\n\t"\
  "fmla v29.4s,v1.4s,v5.s[0]; ldr x10,[x4,#-176]\n\t"\
  "fmla v30.4s,v1.4s,v5.s[1]\n\t"

#define KERNEL_M4N25_TAIL2 \
  "fmov v3.d[1],x10; ldr d4,[x4,#-168]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-160]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]; bfxil x17,x16,#32,#32\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d1,x17\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]; bfxil x20,x19,#32,#32\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "fmov v1.d[1],x20; ldr d2,[x4,#-152]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-144]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x6]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-136]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-128]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-120]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-112]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-104]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-96]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x7]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-88]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-80]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v29.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-72]\n\t"\
  "fmla v30.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-64]\n\t"\
  "fmla v6.4s,v1.4s,v2.s[1]; prfm pldl1keep,[x8]\n\t"\
  "fmla v7.4s,v1.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v8.4s,v1.4s,v2.s[3]\n\t"\
  "fmla v9.4s,v1.4s,v3.s[0]\n\t"\
  "fmla v10.4s,v1.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-56]\n\t"\
  "fmla v11.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-48]\n\t"\
  "fmla v12.4s,v1.4s,v3.s[3]\n\t"\
  "fmla v13.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-40]\n\t"\
  "fmla v14.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-32]\n\t"\
  "fmla v15.4s,v1.4s,v4.s[2]; sub w5,w5,#2\n\t"\
  "fmla v16.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10\n\t"\
  "fmla v17.4s,v1.4s,v2.s[0]\n\t"\
  "fmla v18.4s,v1.4s,v2.s[1]\n\t"\
  "fmla v19.4s,v1.4s,v2.s[2]\n\t"\
  "ldr d4,[x4,#-24]\n\t"\
  "fmla v20.4s,v1.4s,v2.s[3]; ldr x10,[x4,#-16]\n\t"\
  "fmla v21.4s,v1.4s,v3.s[0]\n\t"\
  "fmla v22.4s,v1.4s,v3.s[1]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-8]\n\t"\
  "fmla v23.4s,v1.4s,v3.s[2]; prfm pldl1keep,[x9]\n\t"\
  "fmla v24.4s,v1.4s,v3.s[3]\n\t"\
  "fmla v25.4s,v1.4s,v4.s[0]\n\t"\
  "fmla v26.4s,v1.4s,v4.s[1]\n\t"\
  "fmla v27.4s,v1.4s,v4.s[2]\n\t"\
  "fmla v28.4s,v1.4s,v4.s[3]\n\t"\
  "fmla v29.4s,v1.4s,v5.s[0]\n\t"\
  "fmla v30.4s,v1.4s,v5.s[1]\n\t"

#define KERNEL_M4N25_FIN1 \
  "ldr w16,[x0],#4; ldr q2,[x4]\n\t"\
  "ldr w17,[x1],#4; ldr d3,[x4,#16]\n\t"\
  "ldr w19,[x2],#4; ldr x10,[x4,#24]\n\t"\
  "ldr w20,[x3],#4; orr x16,x16,x17,LSL #32\n\t"\
  "fmov d0,x16; orr x19,x19,x20,LSL #32; fmov v0.d[1],x19\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#32]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#40]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#48]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#56]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#64]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#72]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#80]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#88]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]; add x4,x4,#100\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr s2,[x4,#-4]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v29.4s,v0.4s,v4.s[3]\n\t"\
  "fmla v30.4s,v0.4s,v2.s[0]\n\t"


#define INIT_M4N26 INIT_4V(6, 7, 8, 9) \
  INIT_4V(10, 11, 12, 13) INIT_4V(14, 15, 16, 17)\
  INIT_4V(18, 19, 20, 21) INIT_4V(22, 23, 24, 25)\
  INIT_4V(26, 27, 28, 29) INIT_2V(30, 31)

#define SAVE_M4N26(mode) \
  UNIT_SAVE_M4N4_VC_##mode(6, 7, 8, 9) UNIT_SAVE_M4N4_VC_##mode(10, 11, 12, 13)\
  UNIT_SAVE_M4N4_VC_##mode(14, 15, 16, 17) UNIT_SAVE_M4N4_VC_##mode(18, 19, 20, 21)\
  UNIT_SAVE_M4N4_VC_##mode(22, 23, 24, 25) UNIT_SAVE_M4N4_VC_##mode(26, 27, 28, 29)\
  EDGE_SAVE_M4N1K1_##mode(30) EDGE_SAVE_M4N1K1_##mode(31)

#define KERNEL_M4N26_PRELOAD2 \
  "ldr x16,[x0],#8; ldr x17,[x1],#8; ldr x19,[x2],#8; ldr x20,[x3],#8\n\t"\
  "ldr q2,[x4]; ldr q3,[x4,#16]; ldr x10,[x4,#24]; add x4,x4,#208\n\t"\
  "mov w11,w16; bfi x11,x17,#32,#32; fmov d0,x11\n\t"\
  "mov w11,w19; bfi x11,x20,#32,#32; fmov v0.d[1],x11\n\t"

#define KERNEL_M4N26_MAIN2(ap1, ap2) \
  "fmov v3.d[1],x10; ldr d4,[x4,#-176]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-168]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]; bfxil x17,x16,#32,#32\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d1,x17\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]; ldr x16,[x0],#8\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]; bfxil x20,x19,#32,#32\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "fmov v1.d[1],x20; ldr d2,[x4,#-160]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-152]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x4,#48]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-144]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-136]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]; ldr x17,[x1],#8\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-128]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]; mov w11,w16\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]; bfi x11,x17,#32,#32\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-112]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-104]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x"#ap1",#64]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-96]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]; ldr x19,[x2],#8\n\t"\
  "fmla v29.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-80]\n\t"\
  "fmla v30.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v31.4s,v0.4s,v2.s[1]; prfm pldl1keep,[x4,#96]\n\t"\
  "fmla v6.4s,v1.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d0,x11\n\t"\
  "fmla v7.4s,v1.4s,v2.s[3]; mov w11,w19\n\t"\
  "fmla v8.4s,v1.4s,v3.s[0]; prfm pldl1keep,[x"#ap2",#64]\n\t"\
  "fmla v9.4s,v1.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-64]\n\t"\
  "fmla v10.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v11.4s,v1.4s,v3.s[3]; ldr x20,[x3],#8\n\t"\
  "fmla v12.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-48]\n\t"\
  "fmla v13.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v14.4s,v1.4s,v4.s[2]; sub w5,w5,#2\n\t"\
  "fmla v15.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10\n\t"\
  "fmla v16.4s,v1.4s,v2.s[0]; bfi x11,x20,#32,#32\n\t"\
  "fmla v17.4s,v1.4s,v2.s[1]; cmp w5,#6\n\t"\
  "fmla v18.4s,v1.4s,v2.s[2]\n\t"\
  "ldr d4,[x4,#-32]; fmov v0.d[1],x11\n\t"\
  "fmla v19.4s,v1.4s,v2.s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v20.4s,v1.4s,v3.s[0]; prfm pldl1keep,[x4,#144]\n\t"\
  "fmla v21.4s,v1.4s,v3.s[1]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-16]\n\t"\
  "fmla v22.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-8]\n\t"\
  "fmla v23.4s,v1.4s,v3.s[3]; add x4,x4,#208\n\t"\
  "fmla v24.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v5.d[1],x10; ldr d2,[x4,#-208]\n\t"\
  "fmla v25.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-200]\n\t"\
  "fmla v26.4s,v1.4s,v4.s[2]; prfm pldl1keep,[x4]\n\t"\
  "fmla v27.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v2.d[1],x10\n\t"\
  "fmla v28.4s,v1.4s,v5.s[0]\n\t"\
  "fmla v29.4s,v1.4s,v5.s[1]\n\t"\
  "ldr d3,[x4,#-192]\n\t"\
  "fmla v30.4s,v1.4s,v5.s[2]; ldr x10,[x4,#-184]\n\t"\
  "fmla v31.4s,v1.4s,v5.s[3]\n\t"

#define KERNEL_M4N26_TAIL2 \
  "fmov v3.d[1],x10; ldr d4,[x4,#-176]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-168]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]; bfxil x17,x16,#32,#32\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10; fmov d1,x17\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]; bfxil x20,x19,#32,#32\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "fmov v1.d[1],x20; ldr d2,[x4,#-160]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-152]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x6]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-144]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-136]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-128]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-120]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-112]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]; ldr x10,[x4,#-104]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]; prfm pldl1keep,[x7]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-96]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]; ldr x10,[x4,#-88]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v29.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#-80]\n\t"\
  "fmla v30.4s,v0.4s,v2.s[0]; ldr x10,[x4,#-72]\n\t"\
  "fmla v31.4s,v0.4s,v2.s[1]; prfm pldl1keep,[x8]\n\t"\
  "fmla v6.4s,v1.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v7.4s,v1.4s,v2.s[3]\n\t"\
  "fmla v8.4s,v1.4s,v3.s[0]; prfm pldl1keep,[x9]\n\t"\
  "fmla v9.4s,v1.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-64]\n\t"\
  "fmla v10.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-56]\n\t"\
  "fmla v11.4s,v1.4s,v3.s[3]\n\t"\
  "fmla v12.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#-48]\n\t"\
  "fmla v13.4s,v1.4s,v4.s[1]; ldr x10,[x4,#-40]\n\t"\
  "fmla v14.4s,v1.4s,v4.s[2]; sub w5,w5,#2\n\t"\
  "fmla v15.4s,v1.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10\n\t"\
  "fmla v16.4s,v1.4s,v2.s[0]\n\t"\
  "fmla v17.4s,v1.4s,v2.s[1]\n\t"\
  "fmla v18.4s,v1.4s,v2.s[2]\n\t"\
  "ldr d4,[x4,#-32]\n\t"\
  "fmla v19.4s,v1.4s,v2.s[3]; ldr x10,[x4,#-24]\n\t"\
  "fmla v20.4s,v1.4s,v3.s[0]\n\t"\
  "fmla v21.4s,v1.4s,v3.s[1]\n\t"\
  "fmov v4.d[1],x10; ldr d5,[x4,#-16]\n\t"\
  "fmla v22.4s,v1.4s,v3.s[2]; ldr x10,[x4,#-8]\n\t"\
  "fmla v23.4s,v1.4s,v3.s[3]\n\t"\
  "fmla v24.4s,v1.4s,v4.s[0]\n\t"\
  "fmov v5.d[1],x10\n\t"\
  "fmla v25.4s,v1.4s,v4.s[1]\n\t"\
  "fmla v26.4s,v1.4s,v4.s[2]\n\t"\
  "fmla v27.4s,v1.4s,v4.s[3]\n\t"\
  "fmla v28.4s,v1.4s,v5.s[0]\n\t"\
  "fmla v29.4s,v1.4s,v5.s[1]\n\t"\
  "fmla v30.4s,v1.4s,v5.s[2]\n\t"\
  "fmla v31.4s,v1.4s,v5.s[3]\n\t"

#define KERNEL_M4N26_FIN1 \
  "ldr w16,[x0],#4; ldr q2,[x4]\n\t"\
  "ldr w17,[x1],#4; ldr d3,[x4,#16]\n\t"\
  "ldr w19,[x2],#4; ldr x10,[x4,#24]\n\t"\
  "ldr w20,[x3],#4; orr x16,x16,x17,LSL #32\n\t"\
  "fmov d0,x16; orr x19,x19,x20,LSL #32; fmov v0.d[1],x19\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#32]\n\t"\
  "fmla v6.4s,v0.4s,v2.s[0]; ldr x10,[x4,#40]\n\t"\
  "fmla v7.4s,v0.4s,v2.s[1]\n\t"\
  "fmla v8.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v9.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v10.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v11.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#48]\n\t"\
  "fmla v12.4s,v0.4s,v3.s[2]; ldr x10,[x4,#56]\n\t"\
  "fmla v13.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v14.4s,v0.4s,v4.s[0]\n\t"\
  "fmov v2.d[1],x10; ldr d3,[x4,#64]\n\t"\
  "fmla v15.4s,v0.4s,v4.s[1]; ldr x10,[x4,#72]\n\t"\
  "fmla v16.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v17.4s,v0.4s,v4.s[3]\n\t"\
  "fmov v3.d[1],x10; ldr d4,[x4,#80]\n\t"\
  "fmla v18.4s,v0.4s,v2.s[0]; ldr x10,[x4,#88]\n\t"\
  "fmla v19.4s,v0.4s,v2.s[1]; add x4,x4,#104\n\t"\
  "fmla v20.4s,v0.4s,v2.s[2]\n\t"\
  "fmov v4.d[1],x10\n\t"\
  "fmla v21.4s,v0.4s,v2.s[3]\n\t"\
  "fmla v22.4s,v0.4s,v3.s[0]\n\t"\
  "fmla v23.4s,v0.4s,v3.s[1]\n\t"\
  "ldr d2,[x4,#-8]\n\t"\
  "fmla v24.4s,v0.4s,v3.s[2]\n\t"\
  "fmla v25.4s,v0.4s,v3.s[3]\n\t"\
  "fmla v26.4s,v0.4s,v4.s[0]\n\t"\
  "fmla v27.4s,v0.4s,v4.s[1]\n\t"\
  "fmla v28.4s,v0.4s,v4.s[2]\n\t"\
  "fmla v29.4s,v0.4s,v4.s[3]\n\t"\
  "fmla v30.4s,v0.4s,v2.s[0]\n\t"\
  "fmla v31.4s,v0.4s,v2.s[1]\n\t"

#define FUNC_K1(ndim) \
static inline void sgemm_skinny1_a53_m4n##ndim(\
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
    "cmp w5,#2; b.lt 4f\n\t"\
    KERNEL_M4N##ndim##_PRELOAD2\
    "cmp w5,#6; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    KERNEL_M4N##ndim##_MAIN2(0, 1)\
    KERNEL_M4N##ndim##_MAIN2(2, 3)\
    "b.ge 1b; 2:\n\t"\
    "cmp w5,#4; b.lt 3f\n\t"\
    KERNEL_M4N##ndim##_MAIN2(0, 1)\
    KERNEL_M4N##ndim##_TAIL2\
    "b 4f; 3:\n\t"\
    KERNEL_M4N##ndim##_TAIL2\
    "4:\n\t"\
    "cmp w5,#1; b.lt 6f\n\t"\
    "5:\n\t"\
    KERNEL_M4N##ndim##_FIN1\
    "6:\n\t"\
    INIT_SAVE\
    "cmp %w[c_rowmajor],#0; b.eq 7f\n\t"\
    SAVE_M4N##ndim(CR) "b 8f\n\t"\
    "7:\n\t"\
    SAVE_M4N##ndim(CC)\
    "8:\n\t"\
  ::[a_ptr]"r"(a_ptr), [c_ptr]"r"(c_ptr), [b_scr]"r"(b_scr),\
    [K]"r"(K), [LDA]"r"(LDA), [LDC]"r"(LDC),\
    [beta_addr]"r"(beta_addr), [c_rowmajor]"r"(c_rowmajor)\
  :"cc","memory","x0","x1","x2","x3","x4","x5","x6","x7","x8","x9",\
  "x10","x11","x12","x13","x14","x15","x16","x17","x19","x20",\
  "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13",\
  "v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25",\
  "v26","v27","v28","v29","v30","v31");\
}

FUNC_K1(23)
FUNC_K1(24)
FUNC_K1(25)
FUNC_K1(26)

#define INIT_M1N4 \
  float32x4_t cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = vdupq_n_f32(0.0f);

#define INIT_M1N5 INIT_M1N4 float32x4_t cq5 = cq1;

#define INIT_M1N6 INIT_M1N5 float32x4_t cq6 = cq1;

#define INIT_M1N7 INIT_M1N6 float32x4_t cq7 = cq1;

#define INIT_M1N8 \
  float32x4_t cq1, cq2; cq1 = cq2 = vdupq_n_f32(0.0f);

#define INIT_M1N9 INIT_M1N8 float32x4_t cq3 = cq1;

#define INIT_M1N10 INIT_M1N9 float32x4_t cq4 = cq1;

#define INIT_M1N11 INIT_M1N10 float32x4_t cq5 = cq1;

#define INIT_M1N12 \
  float32x4_t cq1, cq2, cq3; cq1 = cq2 = cq3 = vdupq_n_f32(0.0f);

#define INIT_M1N13 INIT_M1N12 float32x4_t cq4 = cq1;

#define INIT_M1N14 INIT_M1N13 float32x4_t cq5 = cq2;

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

#define ACC_K4M1N5 ACC_K4M1N4 \
  bq1 = vld1q_f32(b_rd + 16); cq5 = vfmaq_f32(cq5, aq1, bq1);

#define ACC_K4M1N6 ACC_K4M1N5 \
  bq2 = vld1q_f32(b_rd + 20); cq6 = vfmaq_f32(cq6, aq1, bq2);

#define ACC_K4M1N7 ACC_K4M1N6 \
  bq3 = vld1q_f32(b_rd + 24); cq7 = vfmaq_f32(cq7, aq1, bq3);

#define ACC_K4M1N8 \
  float32x4_t aq1 = vld1q_f32(a_rd); a_rd += 4;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 0); bq1 = vld1q_f32(b_rd + 8);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 0); bq2 = vld1q_f32(b_rd + 12);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 1); bq1 = vld1q_f32(b_rd + 16);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 1); bq2 = vld1q_f32(b_rd + 20);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 2); bq1 = vld1q_f32(b_rd + 24);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 2); bq2 = vld1q_f32(b_rd + 28);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 3);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 3);

#define ACC_K4M1N9 ACC_K4M1N8 \
  bq1 = vld1q_f32(b_rd + 32); cq3 = vfmaq_f32(cq3, bq1, aq1);

#define ACC_K4M1N10 ACC_K4M1N9 \
  bq2 = vld1q_f32(b_rd + 36); cq4 = vfmaq_f32(cq4, bq2, aq1);

#define ACC_K4M1N11 ACC_K4M1N10 \
  bq1 = vld1q_f32(b_rd + 40); cq5 = vfmaq_f32(cq5, bq1, aq1);

#define ACC_K4M1N12 \
  float32x4_t aq1 = vld1q_f32(a_rd); a_rd += 4;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 0); bq1 = vld1q_f32(b_rd + 12);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 0); bq2 = vld1q_f32(b_rd + 16);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 0); bq3 = vld1q_f32(b_rd + 20);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 1); bq1 = vld1q_f32(b_rd + 24);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 1); bq2 = vld1q_f32(b_rd + 28);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 1); bq3 = vld1q_f32(b_rd + 32);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 2); bq1 = vld1q_f32(b_rd + 36);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 2); bq2 = vld1q_f32(b_rd + 40);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 2); bq3 = vld1q_f32(b_rd + 44);\
  cq1 = vfmaq_laneq_f32(cq1, bq1, aq1, 3);\
  cq2 = vfmaq_laneq_f32(cq2, bq2, aq1, 3);\
  cq3 = vfmaq_laneq_f32(cq3, bq3, aq1, 3);

#define ACC_K4M1N13 ACC_K4M1N12 \
  bq1 = vld1q_f32(b_rd + 48); cq4 = vfmaq_f32(cq4, bq1, aq1);

#define ACC_K4M1N14 ACC_K4M1N13 \
  bq2 = vld1q_f32(b_rd + 52); cq5 = vfmaq_f32(cq5, bq2, aq1);

#define REDUC_N4 \
  cq1 = vaddq_f32(cq1, cq2); cq3 = vaddq_f32(cq3, cq4);\
  cq1 = vaddq_f32(cq1, cq3);

#define REDUC_N5 REDUC_N4 \
  float32x2_t cd1 = vadd_f32(vget_low_f32(cq5), vget_high_f32(cq5));\
  float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);

#define REDUC_N6 REDUC_N5 \
  float32x2_t cd2 = vadd_f32(vget_low_f32(cq6), vget_high_f32(cq6));\
  float cs2 = vget_lane_f32(cd2, 0) + vget_lane_f32(cd2, 1);

#define REDUC_N7 REDUC_N6 \
  float32x2_t cd3 = vadd_f32(vget_low_f32(cq7), vget_high_f32(cq7));\
  float cs3 = vget_lane_f32(cd3, 0) + vget_lane_f32(cd3, 1);

#define REDUC_N8 {}

#define REDUC_N9 \
  float32x2_t cd1 = vadd_f32(vget_low_f32(cq3), vget_high_f32(cq3));\
  float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);

#define REDUC_N10 REDUC_N9 \
  float32x2_t cd2 = vadd_f32(vget_low_f32(cq4), vget_high_f32(cq4));\
  float cs2 = vget_lane_f32(cd2, 0) + vget_lane_f32(cd2, 1);

#define REDUC_N11 REDUC_N10 \
  float32x2_t cd3 = vadd_f32(vget_low_f32(cq5), vget_high_f32(cq5));\
  float cs3 = vget_lane_f32(cd3, 0) + vget_lane_f32(cd3, 1);

#define REDUC_N12 {}

#define REDUC_N13 \
  float32x2_t cd1 = vadd_f32(vget_low_f32(cq4), vget_high_f32(cq4));\
  float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);

#define REDUC_N14 REDUC_N13 \
  float32x2_t cd2 = vadd_f32(vget_low_f32(cq5), vget_high_f32(cq5));\
  float cs2 = vget_lane_f32(cd2, 0) + vget_lane_f32(cd2, 1);

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

#define FUNC_EDGE_K4(ndim) \
static inline void sgemm_skinny1_a53_m1n##ndim(\
  const float * __restrict__ a_rd, const float * __restrict__ b_rd,\
  float * __restrict__ c_ptr, uint32_t k_left, uint32_t LDC,\
  uint8_t c_rowmajor, float beta) {\
  INIT_M1N##ndim\
  for (; k_left > 3; k_left -= 4) {\
    ACC_K4M1N##ndim b_rd += ndim * 4;\
  }\
  REDUC_N##ndim\
  for (; k_left > 0; k_left--) {\
    ACC_K1M1N##ndim b_rd += ndim;\
  }\
  if (c_rowmajor == 0) {\
    SAVE_M1N##ndim(CC)\
  } else {\
    SAVE_M1N##ndim(CR)\
  }\
}

FUNC_EDGE_K4(4)
FUNC_EDGE_K4(5)
FUNC_EDGE_K4(6)
FUNC_EDGE_K4(7)
FUNC_EDGE_K4(8)
FUNC_EDGE_K4(9)
FUNC_EDGE_K4(10)
FUNC_EDGE_K4(11)
FUNC_EDGE_K4(12)
FUNC_EDGE_K4(13)
FUNC_EDGE_K4(14)

#define INIT_M1N15 \
  float32x4_t cq1, cq2, cq3, cq4, cq5, cq6;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = vdupq_n_f32(0.0f);\
  float32x2_t cd1, cd2, cd3;\
  cd1 = cd2 = cd3 = vdup_n_f32(0.0f);

#define INIT_M1N16 \
  float32x4_t cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = vdupq_n_f32(0.0f);

#define INIT_M1N17 INIT_M1N16 float32x2_t cd1 = vdup_n_f32(0.0f);

#define INIT_M1N18 INIT_M1N17 float32x2_t cd2 = vdup_n_f32(0.0f);

#define INIT_M1N19 INIT_M1N18 float32x2_t cd3 = vdup_n_f32(0.0f);

#define INIT_M1N20 INIT_M1N16 float32x4_t cq5 = vdupq_n_f32(0.0f);

#define INIT_M1N21 INIT_M1N20 float32x2_t cd1 = vdup_n_f32(0.0f);

#define INIT_M1N22 INIT_M1N21 float32x2_t cd2 = vdup_n_f32(0.0f);

#define ACC_M1N15K2 \
  float32x2_t ad1 = vld1_f32(a_rd); a_rd += 2;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  cq1 = vfmaq_lane_f32(cq1, bq1, ad1, 0); bq1 = vld1q_f32(b_rd + 12);\
  cq2 = vfmaq_lane_f32(cq2, bq2, ad1, 0); bq2 = vld1q_f32(b_rd + 16);\
  cq3 = vfmaq_lane_f32(cq3, bq3, ad1, 0); bq3 = vld1q_f32(b_rd + 20);\
  cq4 = vfmaq_lane_f32(cq4, bq1, ad1, 1); float32x2_t bd1 = vld1_f32(b_rd + 24);\
  cq5 = vfmaq_lane_f32(cq5, bq2, ad1, 1); float32x2_t bd2 = vld1_f32(b_rd + 26);\
  cq6 = vfmaq_lane_f32(cq6, bq3, ad1, 1); float32x2_t bd3 = vld1_f32(b_rd + 28);\
  cd1 = vfma_f32(cd1, ad1, bd1);\
  cd2 = vfma_f32(cd2, ad1, bd2);\
  cd3 = vfma_f32(cd3, ad1, bd3);

#define ACC_M1N16K2 \
  float32x2_t ad1 = vld1_f32(a_rd); a_rd += 2;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  cq1 = vfmaq_lane_f32(cq1, bq1, ad1, 0); bq1 = vld1q_f32(b_rd + 16);\
  cq2 = vfmaq_lane_f32(cq2, bq2, ad1, 0); bq2 = vld1q_f32(b_rd + 20);\
  cq3 = vfmaq_lane_f32(cq3, bq3, ad1, 0); bq3 = vld1q_f32(b_rd + 24);\
  cq4 = vfmaq_lane_f32(cq4, bq4, ad1, 0); bq4 = vld1q_f32(b_rd + 28);\
  cq1 = vfmaq_lane_f32(cq1, bq1, ad1, 1);\
  cq2 = vfmaq_lane_f32(cq2, bq2, ad1, 1);\
  cq3 = vfmaq_lane_f32(cq3, bq3, ad1, 1);\
  cq4 = vfmaq_lane_f32(cq4, bq4, ad1, 1);

#define ACC_M1N17K2 ACC_M1N16K2 \
  float32x2_t bd1 = vld1_f32(b_rd + 32);\
  cd1 = vfma_f32(cd1, ad1, bd1);

#define ACC_M1N18K2 ACC_M1N17K2 \
  bd1 = vld1_f32(b_rd + 34); cd2 = vfma_f32(cd2, ad1, bd1);

#define ACC_M1N19K2 ACC_M1N18K2 \
  bd1 = vld1_f32(b_rd + 36); cd3 = vfma_f32(cd3, ad1, bd1);

#define ACC_M1N20K2 \
  float32x2_t ad1 = vld1_f32(a_rd); a_rd += 2;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  float32x4_t bq5 = vld1q_f32(b_rd + 16);\
  cq1 = vfmaq_lane_f32(cq1, bq1, ad1, 0); bq1 = vld1q_f32(b_rd + 20);\
  cq2 = vfmaq_lane_f32(cq2, bq2, ad1, 0); bq2 = vld1q_f32(b_rd + 24);\
  cq3 = vfmaq_lane_f32(cq3, bq3, ad1, 0); bq3 = vld1q_f32(b_rd + 28);\
  cq4 = vfmaq_lane_f32(cq4, bq4, ad1, 0); bq4 = vld1q_f32(b_rd + 32);\
  cq5 = vfmaq_lane_f32(cq5, bq5, ad1, 0); bq5 = vld1q_f32(b_rd + 36);\
  cq1 = vfmaq_lane_f32(cq1, bq1, ad1, 1);\
  cq2 = vfmaq_lane_f32(cq2, bq2, ad1, 1);\
  cq3 = vfmaq_lane_f32(cq3, bq3, ad1, 1);\
  cq4 = vfmaq_lane_f32(cq4, bq4, ad1, 1);\
  cq5 = vfmaq_lane_f32(cq5, bq5, ad1, 1);

#define ACC_M1N21K2 ACC_M1N20K2 \
  float32x2_t bd1 = vld1_f32(b_rd + 40); cd1 = vfma_f32(cd1, ad1, bd1);

#define ACC_M1N22K2 ACC_M1N21K2 \
  float32x2_t bd2 = vld1_f32(b_rd + 42); cd2 = vfma_f32(cd2, ad1, bd2);

#define REDUC_M1N15 \
  cq1 = vaddq_f32(cq1, cq4); cq2 = vaddq_f32(cq2, cq5); cq3 = vaddq_f32(cq3, cq6);\
  float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);\
  float cs2 = vget_lane_f32(cd2, 0) + vget_lane_f32(cd2, 1);\
  float cs3 = vget_lane_f32(cd3, 0) + vget_lane_f32(cd3, 1);

#define REDUC_M1N16 {}

#define REDUC_M1N17 float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);

#define REDUC_M1N18 REDUC_M1N17 \
  float cs2 = vget_lane_f32(cd2, 0) + vget_lane_f32(cd2, 1);

#define REDUC_M1N19 REDUC_M1N18 \
  float cs3 = vget_lane_f32(cd3, 0) + vget_lane_f32(cd3, 1);

#define REDUC_M1N20 {}

#define REDUC_M1N21 float cs1 = vget_lane_f32(cd1, 0) + vget_lane_f32(cd1, 1);

#define REDUC_M1N22 REDUC_M1N21 \
  float cs2 = vget_lane_f32(cd2, 0) + vget_lane_f32(cd2, 1);

#define ACC_M1N15K1 \
  float as1 = *a_rd++;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  cq1 = vfmaq_n_f32(cq1, bq1, as1); float bs1 = b_rd[12];\
  cq2 = vfmaq_n_f32(cq2, bq2, as1); float bs2 = b_rd[13];\
  cq3 = vfmaq_n_f32(cq3, bq3, as1); float bs3 = b_rd[14];\
  cs1 += as1 * bs1; cs2 += as1 * bs2; cs3 += as1 * bs3;

#define ACC_M1N16K1 \
  float as1 = *a_rd++;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  cq1 = vfmaq_n_f32(cq1, bq1, as1); cq2 = vfmaq_n_f32(cq2, bq2, as1);\
  cq3 = vfmaq_n_f32(cq3, bq3, as1); cq4 = vfmaq_n_f32(cq4, bq4, as1);

#define ACC_M1N17K1 ACC_M1N16K1 cs1 += as1 * b_rd[16];

#define ACC_M1N18K1 ACC_M1N17K1 cs2 += as1 * b_rd[17];

#define ACC_M1N19K1 ACC_M1N18K1 cs3 += as1 * b_rd[18];

#define ACC_M1N20K1 \
  float as1 = *a_rd++;\
  float32x4_t bq1 = vld1q_f32(b_rd);\
  float32x4_t bq2 = vld1q_f32(b_rd + 4);\
  float32x4_t bq3 = vld1q_f32(b_rd + 8);\
  float32x4_t bq4 = vld1q_f32(b_rd + 12);\
  float32x4_t bq5 = vld1q_f32(b_rd + 16);\
  cq1 = vfmaq_n_f32(cq1, bq1, as1); cq2 = vfmaq_n_f32(cq2, bq2, as1);\
  cq3 = vfmaq_n_f32(cq3, bq3, as1); cq4 = vfmaq_n_f32(cq4, bq4, as1);\
  cq5 = vfmaq_n_f32(cq5, bq5, as1);

#define ACC_M1N21K1 ACC_M1N20K1 cs1 += as1 * b_rd[20];

#define ACC_M1N22K1 ACC_M1N21K1 cs2 += as1 * b_rd[21];

#define SAVE_M1N15(mode) \
  UNIT_SAVE_M1N4_##mode(cq1) UNIT_SAVE_M1N4_##mode(cq2) UNIT_SAVE_M1N4_##mode(cq3)\
  UNIT_SAVE_M1N1_##mode(cs1) UNIT_SAVE_M1N1_##mode(cs2) UNIT_SAVE_M1N1_##mode(cs3)

#define SAVE_M1N16(mode) \
  UNIT_SAVE_M1N4_##mode(cq1) UNIT_SAVE_M1N4_##mode(cq2)\
  UNIT_SAVE_M1N4_##mode(cq3) UNIT_SAVE_M1N4_##mode(cq4)

#define SAVE_M1N17(mode) SAVE_M1N16(mode) UNIT_SAVE_M1N1_##mode(cs1)

#define SAVE_M1N18(mode) SAVE_M1N17(mode) UNIT_SAVE_M1N1_##mode(cs2)

#define SAVE_M1N19(mode) SAVE_M1N18(mode) UNIT_SAVE_M1N1_##mode(cs3)

#define SAVE_M1N20(mode) SAVE_M1N16(mode) UNIT_SAVE_M1N4_##mode(cq5)

#define SAVE_M1N21(mode) SAVE_M1N20(mode) UNIT_SAVE_M1N1_##mode(cs1)

#define SAVE_M1N22(mode) SAVE_M1N21(mode) UNIT_SAVE_M1N1_##mode(cs2)

#define FUNC_EDGE_K2(ndim) \
static inline void sgemm_skinny1_a53_m1n##ndim(\
  const float * __restrict__ a_rd, const float * __restrict__ b_rd,\
  float * __restrict__ c_ptr, uint32_t k_left, uint32_t LDC,\
  uint8_t c_rowmajor, float beta) {\
  INIT_M1N##ndim\
  for (; k_left > 1; k_left -= 2) {\
    ACC_M1N##ndim##K2 b_rd += ndim * 2;\
  }\
  REDUC_M1N##ndim\
  for (; k_left > 0; k_left--) {\
    ACC_M1N##ndim##K1 b_rd += ndim;\
  }\
  if (c_rowmajor == 0) {\
    SAVE_M1N##ndim(CC)\
  } else {\
    SAVE_M1N##ndim(CR)\
  }\
}

FUNC_EDGE_K2(15)
FUNC_EDGE_K2(16)
FUNC_EDGE_K2(17)
FUNC_EDGE_K2(18)
FUNC_EDGE_K2(19)
FUNC_EDGE_K2(20)
FUNC_EDGE_K2(21)
FUNC_EDGE_K2(22)

#define INIT_M1N23 INIT_M1N20 \
  float cs1 = 0.0f, cs2 = 0.0f, cs3 = 0.0f;

#define INIT_M1N24 INIT_M1N20 float32x4_t cq6 = vdupq_n_f32(0.0f);

#define INIT_M1N25 INIT_M1N24 float cs1 = 0.0f;

#define INIT_M1N26 INIT_M1N25 float cs2 = 0.0f;

#define ACC_M1N23K1 ACC_M1N20K1 \
  cs1 += as1 * b_rd[20]; cs2 += as1 * b_rd[21]; cs3 += as1 * b_rd[22];

#define ACC_M1N24K1 ACC_M1N20K1 \
  float32x4_t bq6 = vld1q_f32(b_rd + 20);\
  cq6 = vfmaq_n_f32(cq6, bq6, as1);

#define ACC_M1N25K1 ACC_M1N24K1 cs1 += as1 * b_rd[24];

#define ACC_M1N26K1 ACC_M1N25K1 cs2 += as1 * b_rd[25];

#define SAVE_M1N23(mode) \
  UNIT_SAVE_M1N4_##mode(cq1) UNIT_SAVE_M1N4_##mode(cq2) UNIT_SAVE_M1N4_##mode(cq3)\
  UNIT_SAVE_M1N4_##mode(cq4) UNIT_SAVE_M1N4_##mode(cq5)\
  UNIT_SAVE_M1N1_##mode(cs1) UNIT_SAVE_M1N1_##mode(cs2) UNIT_SAVE_M1N1_##mode(cs3)

#define SAVE_M1N24(mode) \
  UNIT_SAVE_M1N4_##mode(cq1) UNIT_SAVE_M1N4_##mode(cq2) UNIT_SAVE_M1N4_##mode(cq3)\
  UNIT_SAVE_M1N4_##mode(cq4) UNIT_SAVE_M1N4_##mode(cq5) UNIT_SAVE_M1N4_##mode(cq6)

#define SAVE_M1N25(mode) SAVE_M1N24(mode) UNIT_SAVE_M1N1_##mode(cs1)

#define SAVE_M1N26(mode) SAVE_M1N25(mode) UNIT_SAVE_M1N1_##mode(cs2)

#define FUNC_EDGE_K1(ndim) \
static inline void sgemm_skinny1_a53_m1n##ndim(\
  const float * __restrict__ a_rd, const float * __restrict__ b_rd,\
  float * __restrict__ c_ptr, uint32_t k_left, uint32_t LDC,\
  uint8_t c_rowmajor, float beta) {\
  INIT_M1N##ndim\
  for (; k_left > 0; k_left--) {\
    ACC_M1N##ndim##K1 b_rd += ndim;\
  }\
  if (c_rowmajor == 0) {\
    SAVE_M1N##ndim(CC)\
  } else {\
    SAVE_M1N##ndim(CR)\
  }\
}

FUNC_EDGE_K1(23)
FUNC_EDGE_K1(24)
FUNC_EDGE_K1(25)
FUNC_EDGE_K1(26)

#endif
