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


/******************************************************************************
 * File:        NeonI8I32MlaGemmSkinnyDot.h
 * Description: Source code template for NEON mlal 8->32bit GEMM skinny dot
 *              kernels.
 *****************************************************************************/

#include "common/CommonSkinnyDot.h"
#include "arm_neon/NeonIntOpSign.h"

#ifndef INCLUDE_I8I32_MLA_SKINNYDOT
#define INCLUDE_I8I32_MLA_SKINNYDOT

typedef I8 I8I32MLAGEMM_SKINNYDOT_ASCALAR;
typedef I8 I8I32MLAGEMM_SKINNYDOT_BSCALAR;
typedef I32 I8I32MLAGEMM_SKINNYDOT_CSCALAR;

typedef I16 I8I32MLAGEMM_SKINNYDOT_AVEC1;
typedef I16 I8I32MLAGEMM_SKINNYDOT_BVEC1;
typedef I32 I8I32MLAGEMM_SKINNYDOT_CVEC1;

typedef I32X2 I8I32MLAGEMM_SKINNYDOT_AVEC2;
typedef I32X2 I8I32MLAGEMM_SKINNYDOT_BVEC2;
typedef I32X2 I8I32MLAGEMM_SKINNYDOT_CVEC2;

typedef I8X8 I8I32MLAGEMM_SKINNYDOT_AVEC4;
typedef I8X8 I8I32MLAGEMM_SKINNYDOT_BVEC4;
typedef I32X2 I8I32MLAGEMM_SKINNYDOT_CVEC4;

typedef I8X8 I8I32MLAGEMM_SKINNYDOT_AVEC8;
typedef I8X8 I8I32MLAGEMM_SKINNYDOT_BVEC8;
typedef I32X4 I8I32MLAGEMM_SKINNYDOT_CVEC8;

typedef I8X16 I8I32MLAGEMM_SKINNYDOT_AVEC16;
typedef I8X16 I8I32MLAGEMM_SKINNYDOT_BVEC16;
typedef I32X4X2 I8I32MLAGEMM_SKINNYDOT_CVEC16;

#define GEMM_SKINNY_DOT_UNIT_DEDUCE(type, ...)\
  GEMM_SKINNY_DOT_##type##_UNIT(__VA_ARGS__)

GEMM_SKINNY_DOT_UNIT_DEDUCE(CALC, I8I32MLAGEMM, 16) {
  I16X8 low_product = VMULL_I8(VGET_LOW_I8(a_vec), VGET_LOW_I8(b_vec));
#if __aarch64__
  I16X8 high_product = VMULL_HIGH_I8(a_vec, b_vec);
#else
  I16X8 high_product = VMULL_I8(VGET_HIGH_I8(a_vec), VGET_HIGH_I8(b_vec));
#endif
  I32X4X2 ret;
  ret.val[0] = VPADALQ_I16(c_vec.val[0], low_product);
  ret.val[1] = VPADALQ_I16(c_vec.val[1], high_product);
  return ret;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(CALC, I8I32MLAGEMM, 8) {
  I16X8 product = VMULL_I8(a_vec, b_vec);
  return VPADALQ_I16(c_vec, product);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(CALC, I8I32MLAGEMM, 4) {
  I16X8 product = VMULL_I8(a_vec, b_vec);
  return VPADAL_I16(c_vec, VGET_LOW_I16(product));
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(CALC, I8I32MLAGEMM, 2) {
  return VMLA_I32(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(CALC, I8I32MLAGEMM, 1) {
  return c_vec + a_vec * b_vec;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADA, I8I32MLAGEMM, 16) {
#if __aarch64__
  __asm__("prfm pldl1keep,[%0,#80]"::"r"(a_ptr):);
#else
  __asm__("pld [%0,#80]"::"r"(a_ptr):);
#endif
  return VLD1Q_I8(a_ptr);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADA, I8I32MLAGEMM, 8) {
#if __aarch64__
  __asm__("prfm pldl1keep,[%0,#72]"::"r"(a_ptr):);
#else
  __asm__("pld [%0,#72]"::"r"(a_ptr):);
#endif
  return VLD1_I8(a_ptr);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADA, I8I32MLAGEMM, 4) {
#if __aarch64__
  I8X8 ret; /* higher 4 elements not used */
  __asm__("ldr %s0,[%1]; prfm pldl1keep,[%1,#72]":"=w"(ret):"r"(a_ptr):"memory");
  return ret;
#else
  register I8X16 ret __asm("q0"); /* higher 12 elements not used */
  __asm__("vld1.32 {%e0[0]},[%1]; pld [%1,#72]":"=w"(ret):"r"(a_ptr):"memory");
  return VGET_LOW_I8(ret);
#endif
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADA, I8I32MLAGEMM, 2) {
  I32 lo = a_ptr[0];
  I32 hi = a_ptr[1];
  return (I32X2){lo, hi};
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADA, I8I32MLAGEMM, 1) {
  return *a_ptr;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADB, I8I32MLAGEMM, 16) {
  return VLD1Q_I8(b_ptr);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADB, I8I32MLAGEMM, 8) {
  return VLD1_I8(b_ptr);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADB, I8I32MLAGEMM, 4) {
#if __aarch64__
  I8X8 ret; /* higher 4 elements not used */
  __asm__("ldr %s0,[%1]":"=w"(ret):"r"(b_ptr):"memory");
  return ret;
#else
/* armeabi-gcc is always buggy. It always put a 64-bit wide 
 * neon variable into s* register ! 
 * here to use 128-bit wide neon variable to avoid this bug */
  register I8X16 ret __asm("q0"); /* higher 12 elements not used */
  __asm__("vld1.32 {%e0[0]},[%1]":"=w"(ret):"r"(b_ptr):"memory");
  return VGET_LOW_I8(ret);
#endif
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADB, I8I32MLAGEMM, 2) {
  I32 lo = b_ptr[0];
  I32 hi = b_ptr[1];
  return (I32X2){lo, hi};
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(LOADB, I8I32MLAGEMM, 1) {
  return *b_ptr;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(REDUC, I8I32MLAGEMM, 16, 8) {
  return VADDQ_I32(c_vec.val[0], c_vec.val[1]);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(REDUC, I8I32MLAGEMM, 8, 4) {
  return VADD_I32(VGET_LOW_I32(c_vec), VGET_HIGH_I32(c_vec));
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(REDUC, I8I32MLAGEMM, 4, 2) {
  return c_vec;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(REDUC, I8I32MLAGEMM, 2, 1) {
  return VGET_LANE_I32(c_vec, 0) + VGET_LANE_I32(c_vec, 1);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(INITC, I8I32MLAGEMM, 16) {
  I32X4X2 ret;
  ret.val[0] = VDUPQ_N_I32(0);
  ret.val[1] = VDUPQ_N_I32(0);
  return ret;
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(INITC, I8I32MLAGEMM, 8) {
  return VDUPQ_N_I32(0);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(INITC, I8I32MLAGEMM, 4) {
  return VDUP_N_I32(0);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(INITC, I8I32MLAGEMM, 2) {
  return VDUP_N_I32(0);
}

GEMM_SKINNY_DOT_UNIT_DEDUCE(INITC, I8I32MLAGEMM, 1) {
  return 0;
}

#endif