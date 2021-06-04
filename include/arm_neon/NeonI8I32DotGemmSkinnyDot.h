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


/*****************************************************************************
 * File:        NeonI8I32DotGemmSkinnyDot.h
 * Description: Source code template for NEON 8->32bit GEMM skinny dot kernel
 ****************************************************************************/

#include "common/CommonSkinnyDot.h"
#include "arm_neon/NeonIntOpSign.h"

#ifndef INCLUDE_I8I32_DOT_SKINNYDOT
#define INCLUDE_I8I32_DOT_SKINNYDOT

typedef I8 I8I32DOTGEMM_SKINNYDOT_ASCALAR;
typedef I8 I8I32DOTGEMM_SKINNYDOT_BSCALAR;
typedef I32 I8I32DOTGEMM_SKINNYDOT_CSCALAR;

typedef I16 I8I32DOTGEMM_SKINNYDOT_AVEC1;
typedef I16 I8I32DOTGEMM_SKINNYDOT_BVEC1;
typedef I32 I8I32DOTGEMM_SKINNYDOT_CVEC1;

typedef I8X8 I8I32DOTGEMM_SKINNYDOT_AVEC4;
typedef I8X8 I8I32DOTGEMM_SKINNYDOT_BVEC4;
typedef I32X2 I8I32DOTGEMM_SKINNYDOT_CVEC4;

typedef I8X8 I8I32DOTGEMM_SKINNYDOT_AVEC8;
typedef I8X8 I8I32DOTGEMM_SKINNYDOT_BVEC8;
typedef I32X2 I8I32DOTGEMM_SKINNYDOT_CVEC8;

typedef I8X16 I8I32DOTGEMM_SKINNYDOT_AVEC16;
typedef I8X16 I8I32DOTGEMM_SKINNYDOT_BVEC16;
typedef I32X4 I8I32DOTGEMM_SKINNYDOT_CVEC16;

#define GEMM_SKINNY_DOT_UNIT_DEDUCE(TYPE, ...) \
  GEMM_SKINNY_DOT_##TYPE##_UNIT(__VA_ARGS__)

GEMM_SKINNY_DOT_UNIT_DEDUCE(CALC, I8I32DOTGEMM, 16) {
  return VDOTQ_I32(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(CALC, I8I32DOTGEMM, 8) {
  return VDOT_I32(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(CALC, I8I32DOTGEMM, 4) {
  return VDOT_I32(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(CALC, I8I32DOTGEMM, 1) {
  return c_vec + a_vec * b_vec;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADA, I8I32DOTGEMM, 16) {
#if __aarch64__
  __asm__("prfm pldl1keep,[%0,#80]"::"r"(a_ptr):);
#else
  __asm__("pld [%0,#80]"::"r"(a_ptr):);
#endif
  return VLD1Q_I8(a_ptr);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADA, I8I32DOTGEMM, 8) {
#if __aarch64__
  __asm__("prfm pldl1keep,[%0,#72]"::"r"(a_ptr):);
#else
  __asm__("pld [%0,#72]"::"r"(a_ptr):);
#endif
  return VLD1_I8(a_ptr);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADA, I8I32DOTGEMM, 4) {
#if __aarch64__
  I8X8 ret; /* higher 4 elements not used */
  __asm__("ldr %s0,[%1]; prfm pldl1keep,[%1,#72]":"=w"(ret):"r"(a_ptr):"memory");
#else
  register I8X8 ret __asm("d0"); /* higher 4 elements not used */
  __asm__("vld1.32 {%0[0]},[%1]; pld [%1,#72]":"=w"(ret):"r"(a_ptr):"memory");
#endif
  return ret;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADA, I8I32DOTGEMM, 1) {
  return *a_ptr;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADB, I8I32DOTGEMM, 16) {
  return VLD1Q_I8(b_ptr);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADB, I8I32DOTGEMM, 8) {
  return VLD1_I8(b_ptr);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADB, I8I32DOTGEMM, 4) {
#if __aarch64__
  I8X8 ret; /* higher 4 elements not used */
  __asm__("ldr %s0,[%1]":"=w"(ret):"r"(b_ptr):"memory");
#else
  register I8X8 ret __asm("d0"); /* higher 4 elements not used */
  __asm__("vld1.32 {%0[0]},[%1]":"=w"(ret):"r"(b_ptr):"memory");
#endif
  return ret;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADB, I8I32DOTGEMM, 1) {
  return *b_ptr;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(REDUC, I8I32DOTGEMM, 16, 8) {
  return VADD_I32(VGET_LOW_I32(c_vec), VGET_HIGH_I32(c_vec));
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(REDUC, I8I32DOTGEMM, 8, 4) {
  const static I32X2 z0 = {0, 0};
  return VPADD_I32(c_vec, z0);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(REDUC, I8I32DOTGEMM, 4, 1) {
  return VGET_LANE_I32(c_vec, 0);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(INITC, I8I32DOTGEMM, 16) {
  return VDUPQ_N_I32(0);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(INITC, I8I32DOTGEMM, 8) {
  return VDUP_N_I32(0);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(INITC, I8I32DOTGEMM, 4) {
  return VDUP_N_I32(0);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(INITC, I8I32DOTGEMM, 1) {
  return 0;
}

#endif