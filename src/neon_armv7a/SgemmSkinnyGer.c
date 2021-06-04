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

typedef float sgemm_skinnyger_ascalar;
typedef float sgemm_skinnyger_bscalar;
typedef float sgemm_skinnyger_cscalar;

typedef float sgemm_skinnyger_avec1;
typedef float sgemm_skinnyger_bvec1;
typedef float sgemm_skinnyger_cvec1;

typedef float32x2_t sgemm_skinnyger_avec2;
typedef float32x2_t sgemm_skinnyger_bvec2;
typedef float32x2_t sgemm_skinnyger_cvec2;

typedef float32x4_t sgemm_skinnyger_avec4;
typedef float32x4_t sgemm_skinnyger_bvec4;
typedef float32x4_t sgemm_skinnyger_cvec4;

typedef float32x4x2_t sgemm_skinnyger_avec8;
typedef float32x4x2_t sgemm_skinnyger_bvec8;
typedef float32x4x2_t sgemm_skinnyger_cvec8;

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 8, 4, 1) {
  float32x4x2_t ret;
  ret.val[0] = vmlaq_lane_f32(c_vec.val[0], a_vec.val[0], vget_low_f32(b_vec), 0);
  ret.val[1] = vmlaq_lane_f32(c_vec.val[1], a_vec.val[1], vget_low_f32(b_vec), 0);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 8, 4, 2) {
  float32x4x2_t ret;
  ret.val[0] = vmlaq_lane_f32(c_vec.val[0], a_vec.val[0], vget_low_f32(b_vec), 1);
  ret.val[1] = vmlaq_lane_f32(c_vec.val[1], a_vec.val[1], vget_low_f32(b_vec), 1);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 8, 4, 3) {
  float32x4x2_t ret;
  ret.val[0] = vmlaq_lane_f32(c_vec.val[0], a_vec.val[0], vget_high_f32(b_vec), 0);
  ret.val[1] = vmlaq_lane_f32(c_vec.val[1], a_vec.val[1], vget_high_f32(b_vec), 0);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 8, 4, 4) {
  float32x4x2_t ret;
  ret.val[0] = vmlaq_lane_f32(c_vec.val[0], a_vec.val[0], vget_high_f32(b_vec), 1);
  ret.val[1] = vmlaq_lane_f32(c_vec.val[1], a_vec.val[1], vget_high_f32(b_vec), 1);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 8, 2, 1) {
  float32x4x2_t ret;
  ret.val[0] = vmlaq_lane_f32(c_vec.val[0], a_vec.val[0], b_vec, 0);
  ret.val[1] = vmlaq_lane_f32(c_vec.val[1], a_vec.val[1], b_vec, 0);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 8, 2, 2) {
  float32x4x2_t ret;
  ret.val[0] = vmlaq_lane_f32(c_vec.val[0], a_vec.val[0], b_vec, 1);
  ret.val[1] = vmlaq_lane_f32(c_vec.val[1], a_vec.val[1], b_vec, 1);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 8, 1, 1) {
  float32x4x2_t ret;
  ret.val[0] = vmlaq_n_f32(c_vec.val[0], a_vec.val[0], b_vec);
  ret.val[1] = vmlaq_n_f32(c_vec.val[1], a_vec.val[1], b_vec);
  return ret;
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 4, 4, 1) {
  return vmlaq_lane_f32(c_vec, a_vec, vget_low_f32(b_vec), 0);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 4, 4, 2) {
  return vmlaq_lane_f32(c_vec, a_vec, vget_low_f32(b_vec), 1);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 4, 4, 3) {
  return vmlaq_lane_f32(c_vec, a_vec, vget_high_f32(b_vec), 0);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 4, 4, 4) {
  return vmlaq_lane_f32(c_vec, a_vec, vget_high_f32(b_vec), 1);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 4, 2, 1) {
  return vmlaq_lane_f32(c_vec, a_vec, b_vec, 0);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 4, 2, 2) {
  return vmlaq_lane_f32(c_vec, a_vec, b_vec, 1);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 4, 1, 1) {
  return vmlaq_n_f32(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 2, 4, 1) {
  return vmla_lane_f32(c_vec, a_vec, vget_low_f32(b_vec), 0);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 2, 4, 2) {
  return vmla_lane_f32(c_vec, a_vec, vget_low_f32(b_vec), 1);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 2, 4, 3) {
  return vmla_lane_f32(c_vec, a_vec, vget_high_f32(b_vec), 0);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 2, 4, 4) {
  return vmla_lane_f32(c_vec, a_vec, vget_high_f32(b_vec), 1);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 2, 2, 1) {
  return vmla_lane_f32(c_vec, a_vec, b_vec, 0);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 2, 2, 2) {
  return vmla_lane_f32(c_vec, a_vec, b_vec, 1);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 2, 1, 1) {
  return vmla_n_f32(c_vec, a_vec, b_vec);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 1, 4, 1) {
  return c_vec + a_vec * vgetq_lane_f32(b_vec, 0);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 1, 4, 2) {
  return c_vec + a_vec * vgetq_lane_f32(b_vec, 1);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 1, 4, 3) {
  return c_vec + a_vec * vgetq_lane_f32(b_vec, 2);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 1, 4, 4) {
  return c_vec + a_vec * vgetq_lane_f32(b_vec, 3);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 1, 2, 1) {
  return c_vec + a_vec * vget_lane_f32(b_vec, 0);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 1, 2, 2) {
  return c_vec + a_vec * vget_lane_f32(b_vec, 1);
}

GEMM_SKINNY_GER_CALC_UNIT(sgemm, 1, 1, 1) {
  return a_vec * b_vec + c_vec;
}

GEMM_SKINNY_GER_LOADA_UNIT(sgemm, 8) {
  float32x4x2_t ret;
  ret.val[0] = vld1q_f32(a_ptr);
  ret.val[1] = vld1q_f32(a_ptr + 4);
  __asm__("pld [%0,#96]"::"r"(a_ptr):);
  return ret;
}

GEMM_SKINNY_GER_LOADA_UNIT(sgemm, 4) {
  __asm__("pld [%0,#80]"::"r"(a_ptr):);
  return vld1q_f32(a_ptr);
}

GEMM_SKINNY_GER_LOADA_UNIT(sgemm, 2) {
  __asm__("pld [%0,#72]"::"r"(a_ptr):);
  return vld1_f32(a_ptr);
}

GEMM_SKINNY_GER_LOADA_UNIT(sgemm, 1) {
  return *a_ptr;
}

GEMM_SKINNY_GER_LOADC_UNIT(sgemm, 8) {
  float32x4x2_t ret;
  ret.val[0] = vld1q_f32(c_ptr);
  ret.val[1] = vld1q_f32(c_ptr + 4);
  return ret;
}

GEMM_SKINNY_GER_LOADC_UNIT(sgemm, 4) {
  return vld1q_f32(c_ptr);
}

GEMM_SKINNY_GER_LOADC_UNIT(sgemm, 2) {
  return vld1_f32(c_ptr);
}

GEMM_SKINNY_GER_LOADC_UNIT(sgemm, 1) {
  return *c_ptr;
}

GEMM_SKINNY_GER_STOREC_UNIT(sgemm, 8) {
  vst1q_f32(c_ptr, c_vec.val[0]);
  vst1q_f32(c_ptr + 4, c_vec.val[1]);
}

GEMM_SKINNY_GER_STOREC_UNIT(sgemm, 4) {
  vst1q_f32(c_ptr, c_vec);
}

GEMM_SKINNY_GER_STOREC_UNIT(sgemm, 2) {
  vst1_f32(c_ptr, c_vec);
}

GEMM_SKINNY_GER_STOREC_UNIT(sgemm, 1) {
  *c_ptr = c_vec;
}

GEMM_SKINNY_GER_LOADB_UNIT_BROWMAJOR(sgemm, 4) {
  float32x4_t ret = vdupq_n_f32(0);
  float b1 = *b_ptr; b_ptr += ldb;
  float b2 = *b_ptr; b_ptr += ldb;
  float b3 = *b_ptr; b_ptr += ldb;
  float b4 = *b_ptr;
  ret = vsetq_lane_f32(b1, ret, 0);
  ret = vsetq_lane_f32(b2, ret, 1);
  ret = vsetq_lane_f32(b3, ret, 2);
  ret = vsetq_lane_f32(b4, ret, 3);
  return ret;
}

GEMM_SKINNY_GER_LOADB_UNIT_BROWMAJOR(sgemm, 2) {
  float32x2_t ret = vdup_n_f32(0);
  float b1 = *b_ptr;
  float b2 = b_ptr[ldb];
  ret = vset_lane_f32(b1, ret, 0);
  ret = vset_lane_f32(b2, ret, 1);
  return ret;
}

GEMM_SKINNY_GER_LOADB_UNIT_BROWMAJOR(sgemm, 1) {
  return *b_ptr;
}

GEMM_SKINNY_GER_LOADB_UNIT_BCOLMAJOR(sgemm, 4) {
  return vld1q_f32(b_ptr);
}

GEMM_SKINNY_GER_LOADB_UNIT_BCOLMAJOR(sgemm, 2) {
  return vld1_f32(b_ptr);
}

GEMM_SKINNY_GER_LOADB_UNIT_BCOLMAJOR(sgemm, 1) {
  return *b_ptr;
}

GEMM_SKINNY_GER_PARALLEL_FUNC(sgemm, 1, 7, 7, 8192, float, float)
GEMM_SKINNY_GER_PARALLEL_FUNC(sgemm, 2, 7, 7, 8192, float, float)
GEMM_SKINNY_GER_PARALLEL_FUNC(sgemm, 3, 7, 7, 8192, float, float)
GEMM_SKINNY_GER_PARALLEL_FUNC(sgemm, 4, 7, 7, 8192, float, float)
GEMM_SKINNY_GER_PARALLEL_FUNC(sgemm, 5, 7, 7, 8192, float, float)
GEMM_SKINNY_GER_PARALLEL_FUNC(sgemm, 6, 7, 7, 8192, float, float)
GEMM_SKINNY_GER_PARALLEL_FUNC(sgemm, 7, 7, 3, 8192, float, float)
GEMM_SKINNY_GER_PARALLEL_FUNC(sgemm, 8, 7, 3, 8192, float, float)

