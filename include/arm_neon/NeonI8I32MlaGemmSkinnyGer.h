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
 * File:        NeonI8I32MlaGemmSkinnyGer.h
 * Description: Source code template for NEON mlal 8->32bit GEMM skinny ger
 *              kernels.
 *****************************************************************************/

#include "common/CommonSkinnyGer.h"
#include "arm_neon/NeonIntOpSign.h"

#ifndef INCLUDE_I8I32_MLA_SKINNYGER
#define INCLUDE_I8I32_MLA_SKINNYGER

typedef I8 I8I32MLAGEMM_SKINNYGER_ASCALAR;
typedef I8 I8I32MLAGEMM_SKINNYGER_BSCALAR;
typedef I32 I8I32MLAGEMM_SKINNYGER_CSCALAR;

typedef I16 I8I32MLAGEMM_SKINNYGER_AVEC1;
typedef I16 I8I32MLAGEMM_SKINNYGER_BVEC1;
typedef I32 I8I32MLAGEMM_SKINNYGER_CVEC1;

typedef I16X4 I8I32MLAGEMM_SKINNYGER_AVEC4;
typedef I16X4 I8I32MLAGEMM_SKINNYGER_BVEC4;
typedef I32X4 I8I32MLAGEMM_SKINNYGER_CVEC4;

typedef I16X8 I8I32MLAGEMM_SKINNYGER_AVEC8;
typedef I32X4X2 I8I32MLAGEMM_SKINNYGER_CVEC8;

typedef I16X8X2 I8I32MLAGEMM_SKINNYGER_AVEC16;
typedef I32X4X4 I8I32MLAGEMM_SKINNYGER_CVEC16;

#if !__aarch64__
#ifdef VMLAL_HIGH_LANE_I16
#undef VMLAL_HIGH_LANE_I16
#endif
#ifdef VMLAL_HIGH_N_I16
#undef VMLAL_HIGH_N_I16
#endif
#define VMLAL_HIGH_LANE_I16(c, a, b, v) VMLAL_LANE_I16(c, VGET_HIGH_I16(a), b, v)
#define VMLAL_HIGH_N_I16(c, a, b) VMLAL_N_I16(c, VGET_HIGH_I16(a), b)
#endif
#define VMLAL_LOW_LANE_I16(c, a, b, v) VMLAL_LANE_I16(c, VGET_LOW_I16(a), b, v)
#define VMLAL_LOW_N_I16(c, a, b) VMLAL_N_I16(c, VGET_LOW_I16(a), b)

#define GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(type, a, b, c)\
  GEMM_SKINNY_GER_CALC_UNIT(type, a, b, c)

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 16, 4, 1) {
  I32X4X4 ret;
  ret.val[0] = VMLAL_LOW_LANE_I16(c_vec.val[0], a_vec.val[0], b_vec, 0);
  ret.val[1] = VMLAL_HIGH_LANE_I16(c_vec.val[1], a_vec.val[0], b_vec, 0);
  ret.val[2] = VMLAL_LOW_LANE_I16(c_vec.val[2], a_vec.val[1], b_vec, 0);
  ret.val[3] = VMLAL_HIGH_LANE_I16(c_vec.val[3], a_vec.val[1], b_vec, 0);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 16, 4, 2) {
  I32X4X4 ret;
  ret.val[0] = VMLAL_LOW_LANE_I16(c_vec.val[0], a_vec.val[0], b_vec, 1);
  ret.val[1] = VMLAL_HIGH_LANE_I16(c_vec.val[1], a_vec.val[0], b_vec, 1);
  ret.val[2] = VMLAL_LOW_LANE_I16(c_vec.val[2], a_vec.val[1], b_vec, 1);
  ret.val[3] = VMLAL_HIGH_LANE_I16(c_vec.val[3], a_vec.val[1], b_vec, 1);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 16, 4, 3) {
  I32X4X4 ret;
  ret.val[0] = VMLAL_LOW_LANE_I16(c_vec.val[0], a_vec.val[0], b_vec, 2);
  ret.val[1] = VMLAL_HIGH_LANE_I16(c_vec.val[1], a_vec.val[0], b_vec, 2);
  ret.val[2] = VMLAL_LOW_LANE_I16(c_vec.val[2], a_vec.val[1], b_vec, 2);
  ret.val[3] = VMLAL_HIGH_LANE_I16(c_vec.val[3], a_vec.val[1], b_vec, 2);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 16, 4, 4) {
  I32X4X4 ret;
  ret.val[0] = VMLAL_LOW_LANE_I16(c_vec.val[0], a_vec.val[0], b_vec, 3);
  ret.val[1] = VMLAL_HIGH_LANE_I16(c_vec.val[1], a_vec.val[0], b_vec, 3);
  ret.val[2] = VMLAL_LOW_LANE_I16(c_vec.val[2], a_vec.val[1], b_vec, 3);
  ret.val[3] = VMLAL_HIGH_LANE_I16(c_vec.val[3], a_vec.val[1], b_vec, 3);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 16, 1, 1) {
  I32X4X4 ret;
  ret.val[0] = VMLAL_LOW_N_I16(c_vec.val[0], a_vec.val[0], b_vec);
  ret.val[1] = VMLAL_HIGH_N_I16(c_vec.val[1], a_vec.val[0], b_vec);
  ret.val[2] = VMLAL_LOW_N_I16(c_vec.val[2], a_vec.val[1], b_vec);
  ret.val[3] = VMLAL_HIGH_N_I16(c_vec.val[3], a_vec.val[1], b_vec);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 8, 4, 1) {
  I32X4X2 ret;
  ret.val[0] = VMLAL_LOW_LANE_I16(c_vec.val[0], a_vec, b_vec, 0);
  ret.val[1] = VMLAL_HIGH_LANE_I16(c_vec.val[1], a_vec, b_vec, 0);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 8, 4, 2) {
  I32X4X2 ret;
  ret.val[0] = VMLAL_LOW_LANE_I16(c_vec.val[0], a_vec, b_vec, 1);
  ret.val[1] = VMLAL_HIGH_LANE_I16(c_vec.val[1], a_vec, b_vec, 1);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 8, 4, 3) {
  I32X4X2 ret;
  ret.val[0] = VMLAL_LOW_LANE_I16(c_vec.val[0], a_vec, b_vec, 2);
  ret.val[1] = VMLAL_HIGH_LANE_I16(c_vec.val[1], a_vec, b_vec, 2);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 8, 4, 4) {
  I32X4X2 ret;
  ret.val[0] = VMLAL_LOW_LANE_I16(c_vec.val[0], a_vec, b_vec, 3);
  ret.val[1] = VMLAL_HIGH_LANE_I16(c_vec.val[1], a_vec, b_vec, 3);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 8, 1, 1) {
  I32X4X2 ret;
  ret.val[0] = VMLAL_LOW_N_I16(c_vec.val[0], a_vec, b_vec);
  ret.val[1] = VMLAL_HIGH_N_I16(c_vec.val[1], a_vec, b_vec);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 4, 4, 1) {
  return VMLAL_LANE_I16(c_vec, a_vec, b_vec, 0);
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 4, 4, 2) {
  return VMLAL_LANE_I16(c_vec, a_vec, b_vec, 1);
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 4, 4, 3) {
  return VMLAL_LANE_I16(c_vec, a_vec, b_vec, 2);
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 4, 4, 4) {
  return VMLAL_LANE_I16(c_vec, a_vec, b_vec, 3);
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 4, 1, 1) {
  return VMLAL_N_I16(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 1, 4, 1) {
  return c_vec + a_vec * VGET_LANE_I16(b_vec, 0);
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 1, 4, 2) {
  return c_vec + a_vec * VGET_LANE_I16(b_vec, 1);
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 1, 4, 3) {
  return c_vec + a_vec * VGET_LANE_I16(b_vec, 2);
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 1, 4, 4) {
  return c_vec + a_vec * VGET_LANE_I16(b_vec, 3);
}

GEMM_SKINNY_GER_CALC_UNIT_DEDUCE(I8I32MLAGEMM, 1, 1, 1) {
  return c_vec + a_vec * b_vec;
}

#define GEMM_SKINNY_GER_LOADA_UNIT_DEDUCE(type, a)\
  GEMM_SKINNY_GER_LOADA_UNIT(type, a)

GEMM_SKINNY_GER_LOADA_UNIT_DEDUCE(I8I32MLAGEMM, 16) {
  I8X16 ld = VLD1Q_I8(a_ptr);
  I16X8X2 ret;
  ret.val[0] = VMOVL_I8(VGET_LOW_I8(ld));
#if __aarch64__
  ret.val[1] = VMOVL_HIGH_I8(ld);
  __asm__("prfm pldl1keep,[%0,#80]"::"r"(a_ptr):);
#else
  ret.val[1] = VMOVL_I8(VGET_HIGH_I8(ld));
  __asm__("pld [%0,#80]"::"r"(a_ptr):);
#endif
  return ret;
}

GEMM_SKINNY_GER_LOADA_UNIT_DEDUCE(I8I32MLAGEMM, 8) {
  I8X8 t1 = VLD1_I8(a_ptr);
#if __aarch64__
  __asm__("prfm pldl1keep,[%0,#72]"::"r"(a_ptr):);
#else
  __asm__("pld [%0,#72]"::"r"(a_ptr):);
#endif
  return VMOVL_I8(t1);
}

GEMM_SKINNY_GER_LOADA_UNIT_DEDUCE(I8I32MLAGEMM, 4) {
#if __aarch64__
  I16X4 ret;
  __asm__("ldr %s0,[%1]; "ISHLL" %0.8h,%0.8b,#0; prfm pldl1keep,[%1,#72]\n\t"
    :"=w"(ret):"r"(a_ptr):"memory","cc");
  return ret;
#else
  I16X8 ret;
  __asm__("vld1.32 {d0[0]},[%1]; "ASM_VMOVL_I8" %q0,d0; pld [%1,#68]\n\t"
    :"=w"(ret):"r"(a_ptr):"memory","cc","d0");
  return VGET_LOW_I16(ret);
#endif
}

GEMM_SKINNY_GER_LOADA_UNIT_DEDUCE(I8I32MLAGEMM, 1) {
  return *a_ptr;
}

#define GEMM_SKINNY_GER_LOADC_UNIT_DEDUCE(type, a)\
  GEMM_SKINNY_GER_LOADC_UNIT(type, a)

GEMM_SKINNY_GER_LOADC_UNIT_DEDUCE(I8I32MLAGEMM, 16) {
  I32X4X4 ret;
  ret.val[0] = VLD1Q_I32(c_ptr);
  ret.val[1] = VLD1Q_I32(c_ptr + 4);
  ret.val[2] = VLD1Q_I32(c_ptr + 8);
  ret.val[3] = VLD1Q_I32(c_ptr + 12);
  return ret;
}

GEMM_SKINNY_GER_LOADC_UNIT_DEDUCE(I8I32MLAGEMM, 8) {
  I32X4X2 ret;
  ret.val[0] = VLD1Q_I32(c_ptr);
  ret.val[1] = VLD1Q_I32(c_ptr + 4);
  return ret;
}

GEMM_SKINNY_GER_LOADC_UNIT_DEDUCE(I8I32MLAGEMM, 4) {
  return VLD1Q_I32(c_ptr);
}

GEMM_SKINNY_GER_LOADC_UNIT_DEDUCE(I8I32MLAGEMM, 1) {
  return *c_ptr;
}

#define GEMM_SKINNY_GER_STOREC_UNIT_DEDUCE(type, c)\
  GEMM_SKINNY_GER_STOREC_UNIT(type, c)

GEMM_SKINNY_GER_STOREC_UNIT_DEDUCE(I8I32MLAGEMM, 16) {
  VST1Q_I32(c_ptr, c_vec.val[0]);
  VST1Q_I32(c_ptr + 4, c_vec.val[1]);
  VST1Q_I32(c_ptr + 8, c_vec.val[2]);
  VST1Q_I32(c_ptr + 12, c_vec.val[3]);
}

GEMM_SKINNY_GER_STOREC_UNIT_DEDUCE(I8I32MLAGEMM, 8) {
  VST1Q_I32(c_ptr, c_vec.val[0]);
  VST1Q_I32(c_ptr + 4, c_vec.val[1]);
}

GEMM_SKINNY_GER_STOREC_UNIT_DEDUCE(I8I32MLAGEMM, 4) {
  VST1Q_I32(c_ptr, c_vec);
}

GEMM_SKINNY_GER_STOREC_UNIT_DEDUCE(I8I32MLAGEMM, 1) {
  *c_ptr = c_vec;
}

#define GEMM_SKINNY_GER_LOADB_UNIT_DEDUCE(mode, type, b)\
  GEMM_SKINNY_GER_LOADB_UNIT_##mode(type, b)

GEMM_SKINNY_GER_LOADB_UNIT_DEDUCE(BROWMAJOR, I8I32MLAGEMM, 4) {
  I16X4 ret = VDUP_N_I16(0);
  I16 r1 = *b_ptr; b_ptr += ldb;
  I16 r2 = *b_ptr; b_ptr += ldb;
  I16 r3 = *b_ptr; b_ptr += ldb;
  I16 r4 = *b_ptr;
  ret = VSET_LANE_I16(r1, ret, 0);
  ret = VSET_LANE_I16(r2, ret, 1);
  ret = VSET_LANE_I16(r3, ret, 2);
  ret = VSET_LANE_I16(r4, ret, 3);
  return ret;
}

GEMM_SKINNY_GER_LOADB_UNIT_DEDUCE(BROWMAJOR, I8I32MLAGEMM, 1) {
  return *b_ptr;
}

GEMM_SKINNY_GER_LOADB_UNIT_DEDUCE(BCOLMAJOR, I8I32MLAGEMM, 4) {
#if __aarch64__
  I16X4 ret;
  __asm__("ldr %s0,[%1]; "ISHLL" %0.8h,%0.8b,#0\n\t"
    :"=w"(ret):"r"(b_ptr):"memory","cc");
  return ret;
#else
  I16X8 ret;
  __asm__("vld1.32 {d0[0]},[%1]; "ASM_VMOVL_I8" %q0,d0\n\t"
    :"=w"(ret):"r"(b_ptr):"memory","cc","d0");
  return VGET_LOW_I16(ret);
#endif
}

GEMM_SKINNY_GER_LOADB_UNIT_DEDUCE(BCOLMAJOR, I8I32MLAGEMM, 1) {
  return *b_ptr;
}

#endif
