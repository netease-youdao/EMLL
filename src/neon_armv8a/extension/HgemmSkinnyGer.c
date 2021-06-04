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
#include "common/CommonSkinnyGer.h"

#include <arm_neon.h>

typedef float16_t hgemm_skinnyger_ascalar;
typedef float16_t hgemm_skinnyger_bscalar;
typedef float16_t hgemm_skinnyger_cscalar;

typedef float16_t hgemm_skinnyger_avec1;
typedef float16_t hgemm_skinnyger_bvec1;
typedef float16_t hgemm_skinnyger_cvec1;

typedef float16x4_t hgemm_skinnyger_avec4;
typedef float16x4_t hgemm_skinnyger_bvec4;
typedef float16x4_t hgemm_skinnyger_cvec4;

typedef float16x8_t hgemm_skinnyger_avec8;
typedef float16x8_t hgemm_skinnyger_bvec8;
typedef float16x8_t hgemm_skinnyger_cvec8;

typedef float16x8x2_t hgemm_skinnyger_avec16;
typedef float16x8x2_t hgemm_skinnyger_bvec16;
typedef float16x8x2_t hgemm_skinnyger_cvec16;

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 16, 4, 1) {
  float16x8x2_t ret;
  ret.val[0] = vfmaq_lane_f16(c_vec.val[0], a_vec.val[0], b_vec, 0);
  ret.val[1] = vfmaq_lane_f16(c_vec.val[1], a_vec.val[1], b_vec, 0);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 16, 4, 2) {
  float16x8x2_t ret;
  ret.val[0] = vfmaq_lane_f16(c_vec.val[0], a_vec.val[0], b_vec, 1);
  ret.val[1] = vfmaq_lane_f16(c_vec.val[1], a_vec.val[1], b_vec, 1);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 16, 4, 3) {
  float16x8x2_t ret;
  ret.val[0] = vfmaq_lane_f16(c_vec.val[0], a_vec.val[0], b_vec, 2);
  ret.val[1] = vfmaq_lane_f16(c_vec.val[1], a_vec.val[1], b_vec, 2);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 16, 4, 4) {
  float16x8x2_t ret;
  ret.val[0] = vfmaq_lane_f16(c_vec.val[0], a_vec.val[0], b_vec, 3);
  ret.val[1] = vfmaq_lane_f16(c_vec.val[1], a_vec.val[1], b_vec, 3);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 16, 1, 1) {
  float16x8x2_t ret;
  ret.val[0] = vfmaq_n_f16(c_vec.val[0], a_vec.val[0], b_vec);
  ret.val[1] = vfmaq_n_f16(c_vec.val[1], a_vec.val[1], b_vec);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 8, 4, 1) {
  return vfmaq_lane_f16(c_vec, a_vec, b_vec, 0);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 8, 4, 2) {
  return vfmaq_lane_f16(c_vec, a_vec, b_vec, 1);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 8, 4, 3) {
  return vfmaq_lane_f16(c_vec, a_vec, b_vec, 2);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 8, 4, 4) {
  return vfmaq_lane_f16(c_vec, a_vec, b_vec, 3);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 8, 1, 1) {
  return vfmaq_n_f16(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 4, 4, 1) {
  return vfma_lane_f16(c_vec, a_vec, b_vec, 0);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 4, 4, 2) {
  return vfma_lane_f16(c_vec, a_vec, b_vec, 1);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 4, 4, 3) {
  return vfma_lane_f16(c_vec, a_vec, b_vec, 2);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 4, 4, 4) {
  return vfma_lane_f16(c_vec, a_vec, b_vec, 3);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 4, 1, 1) {
  return vfma_n_f16(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 1, 4, 1) {
  return c_vec + a_vec * vget_lane_f16(b_vec, 0);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 1, 4, 2) {
  return c_vec + a_vec * vget_lane_f16(b_vec, 1);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 1, 4, 3) {
  return c_vec + a_vec * vget_lane_f16(b_vec, 2);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 1, 4, 4) {
  return c_vec + a_vec * vget_lane_f16(b_vec, 3);
}

GEMM_SKINNY_GER_CALC_UNIT(hgemm, 1, 1, 1) {
  return c_vec + a_vec * b_vec;
}

GEMM_SKINNY_GER_LOADA_UNIT(hgemm, 16) {
  float16x8x2_t ret;
  ret.val[0] = vld1q_f16(a_ptr);
  ret.val[1] = vld1q_f16(a_ptr + 8);
  __asm__("prfm pldl1keep,[%0,#96]"::"r"(a_ptr):);
  return ret;
}

GEMM_SKINNY_GER_LOADA_UNIT(hgemm, 8) {
  __asm__("prfm pldl1keep,[%0,#80]"::"r"(a_ptr):);
  return vld1q_f16(a_ptr);
}

GEMM_SKINNY_GER_LOADA_UNIT(hgemm, 4) {
  return vld1_f16(a_ptr);
}

GEMM_SKINNY_GER_LOADA_UNIT(hgemm, 1) {
  return *a_ptr;
}

GEMM_SKINNY_GER_LOADC_UNIT(hgemm, 16) {
  float16x8x2_t ret;
  ret.val[0] = vld1q_f16(c_ptr);
  ret.val[1] = vld1q_f16(c_ptr + 8);
  return ret;
}

GEMM_SKINNY_GER_LOADC_UNIT(hgemm, 8) {
  return vld1q_f16(c_ptr);
}

GEMM_SKINNY_GER_LOADC_UNIT(hgemm, 4) {
  return vld1_f16(c_ptr);
}

GEMM_SKINNY_GER_LOADC_UNIT(hgemm, 1) {
  return *c_ptr;
}

GEMM_SKINNY_GER_STOREC_UNIT(hgemm, 16) {
  vst1q_f16(c_ptr, c_vec.val[0]);
  vst1q_f16(c_ptr + 8, c_vec.val[1]);
}

GEMM_SKINNY_GER_STOREC_UNIT(hgemm, 8) {
  vst1q_f16(c_ptr, c_vec);
}

GEMM_SKINNY_GER_STOREC_UNIT(hgemm, 4) {
  vst1_f16(c_ptr, c_vec);
}

GEMM_SKINNY_GER_STOREC_UNIT(hgemm, 1) {
  *c_ptr = c_vec;
}

GEMM_SKINNY_GER_LOADB_UNIT_BROWMAJOR(hgemm, 4) {
  float16x4_t ret = vdup_n_f16(0);
  float16_t b1 = *b_ptr; b_ptr += ldb;
  float16_t b2 = *b_ptr; b_ptr += ldb;
  float16_t b3 = *b_ptr; b_ptr += ldb;
  float16_t b4 = *b_ptr;
  ret = vset_lane_f16(b1, ret, 0);
  ret = vset_lane_f16(b2, ret, 1);
  ret = vset_lane_f16(b3, ret, 2);
  ret = vset_lane_f16(b4, ret, 3);
  return ret;
}

GEMM_SKINNY_GER_LOADB_UNIT_BROWMAJOR(hgemm, 1) {
  return *b_ptr;
}

GEMM_SKINNY_GER_LOADB_UNIT_BCOLMAJOR(hgemm, 4) {
  return vld1_f16(b_ptr);
}

GEMM_SKINNY_GER_LOADB_UNIT_BCOLMAJOR(hgemm, 1) {
  return *b_ptr;
}

GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 1, 5, 29, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 2, 5, 29, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 3, 5, 29, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 4, 5, 29, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 5, 5, 29, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 6, 5, 29, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 7, 5, 29, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 8, 5, 13, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 9, 5, 13, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 10, 5, 13, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 11, 5, 13, 16384, float16_t, float16_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(hgemm, 12, 5, 13, 16384, float16_t, float16_t)

