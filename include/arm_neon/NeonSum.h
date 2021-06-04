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
 * File:        NeonSum.h
 * Description: Sum functions based on ARM NEON instructions.
 ****************************************************************************/

#include <stdint.h>
#include <stdbool.h>
#include <arm_neon.h>

#ifndef INCLUDE_NEON_SUM
#define INCLUDE_NEON_SUM

static inline int16x8_t vaddl_low_s8(int8x16_t v1, int8x16_t v2) {
  return vaddl_s8(vget_low_s8(v1), vget_low_s8(v2));
}

static inline int32x4_t vaddl_low_s16(int16x8_t v1, int16x8_t v2) {
  return vaddl_s16(vget_low_s16(v1), vget_low_s16(v2));
}

static inline uint16x8_t vaddl_low_u8(uint8x16_t v1, uint8x16_t v2) {
  return vaddl_u8(vget_low_u8(v1), vget_low_u8(v2));
}

static inline uint32x4_t vaddl_low_u16(uint16x8_t v1, uint16x8_t v2) {
  return vaddl_u16(vget_low_u16(v1), vget_low_u16(v2));
}

#if !__aarch64__
static inline int16x8_t vaddl_high_s8(int8x16_t v1, int8x16_t v2) {
  return vaddl_s8(vget_high_s8(v1), vget_high_s8(v2));
}

static inline int32x4_t vaddl_high_s16(int16x8_t v1, int16x8_t v2) {
  return vaddl_s16(vget_high_s16(v1), vget_high_s16(v2));
}

static inline uint16x8_t vaddl_high_u8(uint8x16_t v1, uint8x16_t v2) {
  return vaddl_u8(vget_high_u8(v1), vget_high_u8(v2));
}

static inline uint32x4_t vaddl_high_u16(uint16x8_t v1, uint16x8_t v2) {
  return vaddl_u16(vget_high_u16(v1), vget_high_u16(v2));
}

static inline int32x4_t vaddw_high_s16(int32x4_t qv, int16x8_t dv) {
  return vaddw_s16(qv, vget_high_s16(dv));
}

static inline uint32x4_t vaddw_high_u16(uint32x4_t qv, uint16x8_t dv) {
  return vaddw_u16(qv, vget_high_u16(dv));
}
#endif

static inline void pref_src(const void *dat) {
#if __aarch64__
  __asm__("prfm pldl1keep,[%0,#64]\n\t"::"r"(dat):);
#else
  __asm__("pld [%0,#64]\n\t"::"r"(dat):);
#endif
}

/*****************************************************************************
 * Template:    NEON_I8I32_SUM
 * Description: Function template for NEON-based summing operation of a matrix.
 * Template Parameters: sign_short: the integer sign char in the name of
 *                          NEON intrinsics. Please use 's' for signed int
 *                          and 'u' for unsigned int.
 *                      sign_scalar: the string showing integer sign in the
 *                           name of integer type. Please use "int" for
 *                           signed int and "uint" for unsigned int.
 * Function Parameters: src: the address of input matrix.
 *                      dst: the address of output vector.
 *                      dim1: the length of major dimension of input matrix.
 *                      dim2: the length of minor dimension of input matrix.
 *                      (the major dimension is the vertical one for column-
 *                       major matrix, or the horizontal one for row-major
 *                       matrix)
 *                      direction: the direction of summing
 *                                 0: sum along the minor dimension,
 *                                    output_vector_size == dim1;
 *                                 1: sum along the major dimension,
 *                                    output_vector_size == dim2.
 ****************************************************************************/
#define NEON_I8I32_SUM(sign_short, sign_scalar) \
void sign_short##8##sign_short##32##_sum(const sign_scalar##8_t *src,\
  sign_scalar##32_t *dst, uint32_t dim1, uint32_t dim2, uint8_t direction) {\
\
  if (direction == 0) {/* output_size = dim1 */\
    /* first zero output */\
    const sign_scalar##32x4_t z1 = vdupq_n_##sign_short##32(0);\
    uint32_t dim1_left = dim1;\
    sign_scalar##32_t *dst1 = dst;\
    for (; dim1_left > 3; dim1_left -= 4) {\
      vst1q_##sign_short##32(dst1, z1); dst1 += 4;\
    }\
    for (; dim1_left > 0; dim1_left--) {\
      *dst1 = 0; dst1++;\
    }\
    /* then accumulate */\
    const sign_scalar##8_t *src1 = src;\
    uint32_t dim2_left = dim2;\
    for (; dim2_left > 3; dim2_left -= 4) {\
      const sign_scalar##8_t *src_l1 = src1;\
      const sign_scalar##8_t *src_l2 = src1 + dim1;\
      const sign_scalar##8_t *src_l3 = src1 + dim1 * 2;\
      const sign_scalar##8_t *src_l4 = src_l2 + dim1 * 2;\
      src1 = src_l3 + dim1 * 2;\
      sign_scalar##32_t *dst1 = dst;\
      dim1_left = dim1;\
      for (; dim1_left > 15; dim1_left -= 16) {\
        sign_scalar##8x16_t q1 = vld1q_##sign_short##8(src_l1);\
        src_l1 += 16; pref_src(src_l1);\
        sign_scalar##8x16_t q2 = vld1q_##sign_short##8(src_l2);\
        src_l2 += 16; pref_src(src_l2);\
        sign_scalar##8x16_t q3 = vld1q_##sign_short##8(src_l3);\
        src_l3 += 16; pref_src(src_l3);\
        sign_scalar##8x16_t q4 = vld1q_##sign_short##8(src_l4);\
        src_l4 += 16; pref_src(src_l4);\
        sign_scalar##16x8_t m1 = vaddl_low_##sign_short##8(q1, q2);\
        sign_scalar##16x8_t m2 = vaddl_high_##sign_short##8(q1, q2);\
        sign_scalar##16x8_t m3 = vaddl_low_##sign_short##8(q3, q4);\
        sign_scalar##16x8_t m4 = vaddl_high_##sign_short##8(q3, q4);\
        sign_scalar##32x4_t c1 = vld1q_##sign_short##32(dst1);\
        sign_scalar##32x4_t c2 = vld1q_##sign_short##32(dst1 + 4);\
        sign_scalar##32x4_t c3 = vld1q_##sign_short##32(dst1 + 8);\
        sign_scalar##32x4_t c4 = vld1q_##sign_short##32(dst1 + 12);\
        m1 = vaddq_##sign_short##16(m1, m3);\
        m2 = vaddq_##sign_short##16(m2, m4);\
        c1 = vaddw_##sign_short##16(c1, vget_low_##sign_short##16(m1));\
        c2 = vaddw_high_##sign_short##16(c2, m1);\
        c3 = vaddw_##sign_short##16(c3, vget_low_##sign_short##16(m2));\
        c4 = vaddw_high_##sign_short##16(c4, m2);\
        vst1q_##sign_short##32(dst1, c1);\
        vst1q_##sign_short##32(dst1 + 4, c2);\
        vst1q_##sign_short##32(dst1 + 8, c3);\
        vst1q_##sign_short##32(dst1 + 12, c4); dst1 += 16;\
      }\
      if (dim1_left > 7) {\
        sign_scalar##8x8_t d1 = vld1_##sign_short##8(src_l1); src_l1 += 8;\
        sign_scalar##8x8_t d2 = vld1_##sign_short##8(src_l2); src_l2 += 8;\
        sign_scalar##8x8_t d3 = vld1_##sign_short##8(src_l3); src_l3 += 8;\
        sign_scalar##8x8_t d4 = vld1_##sign_short##8(src_l4); src_l4 += 8;\
        sign_scalar##32x4_t c1 = vld1q_##sign_short##32(dst1);\
        sign_scalar##32x4_t c2 = vld1q_##sign_short##32(dst1 + 4);\
        sign_scalar##16x8_t m1 = vaddl_##sign_short##8(d1, d2);\
        sign_scalar##16x8_t m2 = vaddl_##sign_short##8(d3, d4);\
        m1 = vaddq_##sign_short##16(m1, m2);\
        c1 = vaddw_##sign_short##16(c1, vget_low_##sign_short##16(m1));\
        c2 = vaddw_high_##sign_short##16(c2, m1);\
        vst1q_##sign_short##32(dst1, c1);\
        vst1q_##sign_short##32(dst1 + 4, c2); dst1 += 8;\
        dim1_left -= 8;\
      }\
      for (; dim1_left > 0; dim1_left--) {\
        sign_scalar##16_t s1 = *src_l1++;\
        sign_scalar##16_t s2 = *src_l2++;\
        sign_scalar##16_t s3 = *src_l3++;\
        sign_scalar##16_t s4 = *src_l4++;\
        sign_scalar##32_t cs1 = *dst1;\
        s1 += s2; s3 += s4; s1 += s3; cs1 += s1;\
        *dst1 = cs1; dst1++;\
      }\
    }\
    for (; dim2_left > 0; dim2_left--) {\
      sign_scalar##32_t *dst1 = dst;\
      dim1_left = dim1;\
      for (; dim1_left > 15; dim1_left -= 16) {\
        sign_scalar##8x8_t d1 = vld1_##sign_short##8(src1);\
        sign_scalar##8x8_t d2 = vld1_##sign_short##8(src1 + 8); src1 += 16;\
        sign_scalar##16x8_t q1 = vmovl_##sign_short##8(d1);\
        sign_scalar##16x8_t q2 = vmovl_##sign_short##8(d2);\
        sign_scalar##32x4_t c1 = vld1q_##sign_short##32(dst1);\
        sign_scalar##32x4_t c2 = vld1q_##sign_short##32(dst1 + 4);\
        sign_scalar##32x4_t c3 = vld1q_##sign_short##32(dst1 + 8);\
        sign_scalar##32x4_t c4 = vld1q_##sign_short##32(dst1 + 12);\
        c1 = vaddw_##sign_short##16(c1, vget_low_##sign_short##16(q1));\
        c2 = vaddw_high_##sign_short##16(c2, q1);\
        c3 = vaddw_##sign_short##16(c3, vget_low_##sign_short##16(q2));\
        c4 = vaddw_high_##sign_short##16(c4, q2);\
        vst1q_##sign_short##32(dst1, c1);\
        vst1q_##sign_short##32(dst1 + 4, c2);\
        vst1q_##sign_short##32(dst1 + 8, c3);\
        vst1q_##sign_short##32(dst1 + 12, c4);\
        dst1 += 16;\
      }\
      if (dim1_left > 7) {\
        sign_scalar##8x8_t d1 = vld1_##sign_short##8(src1); src1 += 8;\
        sign_scalar##16x8_t q1 = vmovl_##sign_short##8(d1);\
        sign_scalar##32x4_t c1 = vld1q_##sign_short##32(dst1);\
        sign_scalar##32x4_t c2 = vld1q_##sign_short##32(dst1 + 4);\
        c1 = vaddw_##sign_short##16(c1, vget_low_##sign_short##16(q1));\
        c2 = vaddw_high_##sign_short##16(c2, q1);\
        vst1q_##sign_short##32(dst1, c1);\
        vst1q_##sign_short##32(dst1 + 4, c2);\
        dst1 += 8; dim1_left -= 8;\
      }\
      for (; dim1_left > 0; dim1_left--) {\
        *dst1 += *src1; src1++; dst1++;\
      }\
    }\
  } else {/* output size = dim2 */\
    const sign_scalar##8_t *src1 = src;\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; dim2_pos++) {\
      sign_scalar##32x4_t cq1 = vdupq_n_##sign_short##32(0);\
      uint32_t dim1_left = dim1;\
      for (; dim1_left > 15; dim1_left -= 16) {\
        sign_scalar##8x16_t aq1 = vld1q_##sign_short##8(src1); src1 += 16;\
        sign_scalar##16x8_t tq1 = vpaddlq_##sign_short##8(aq1);\
        cq1 = vpadalq_##sign_short##16(cq1, tq1);\
      }\
      sign_scalar##32x2_t cd1 = vadd_##sign_short##32(\
        vget_low_##sign_short##32(cq1), vget_high_##sign_short##32(cq1));\
      if (dim1_left > 7) {\
        sign_scalar##8x8_t ad1 = vld1_##sign_short##8(src1); src1 += 8;\
        sign_scalar##16x4_t td1 = vpaddl_##sign_short##8(ad1);\
        cd1 = vpadal_##sign_short##16(cd1, td1);\
        dim1_left -= 8;\
      }\
      sign_scalar##32_t cs1 = vget_lane_##sign_short##32(\
        vpadd_##sign_short##32(cd1, cd1), 0);\
      for (; dim1_left > 0; dim1_left--) {\
        cs1 += *src1; src1++;\
      }\
      dst[dim2_pos] = cs1;\
    }\
  }\
}

static inline int32x4_t vmull_low_s16(int16x8_t a, int16x8_t b) {
  return vmull_s16(vget_low_s16(a), vget_low_s16(b));
}

static inline uint32x4_t vmull_low_u16(uint16x8_t a, uint16x8_t b) {
  return vmull_u16(vget_low_u16(a), vget_low_u16(b));
}

#if !__aarch64__
static inline int32x4_t vmull_high_s16(int16x8_t a, int16x8_t b) {
  return vmull_s16(vget_high_s16(a), vget_high_s16(b));
}

static inline uint32x4_t vmull_high_u16(uint16x8_t a, uint16x8_t b) {
  return vmull_u16(vget_high_u16(a), vget_high_u16(b));
}
#endif

#define NEON_I16_SUMSQUARE(sign_short, sign_scalar) \
void sign_short##16_sumsquare(const sign_scalar##16_t *dat,\
  sign_scalar##32_t *sum, sign_scalar##64_t *sumsquare, uint32_t size) {\
\
  sign_scalar##32x4_t sum1 = vdupq_n_##sign_short##32(0);\
  sign_scalar##32x4_t sum2 = vdupq_n_##sign_short##32(0);\
  sign_scalar##64x2_t sumsq1 = vdupq_n_##sign_short##64(0);\
  sign_scalar##64x2_t sumsq2 = vdupq_n_##sign_short##64(0);\
  sign_scalar##64x2_t sumsq3 = vdupq_n_##sign_short##64(0);\
  sign_scalar##64x2_t sumsq4 = vdupq_n_##sign_short##64(0);\
\
  if (!sumsquare) {\
    if (sum) {\
      for (; size > 15; size -= 16) {\
        sign_scalar##16x8_t l1 = vld1q_##sign_short##16(dat);\
        sign_scalar##16x8_t l2 = vld1q_##sign_short##16(dat + 8); dat += 16;\
        sum1 = vpadalq_##sign_short##16(sum1, l1);\
        sum2 = vpadalq_##sign_short##16(sum2, l2);\
      }\
      sum1 = vaddq_##sign_short##32(sum1, sum2);\
      if (size > 7) {\
        sign_scalar##16x8_t l1 = vld1q_##sign_short##16(dat); dat += 8;\
        sum1 = vpadalq_##sign_short##16(sum1, l1);\
        size -= 8;\
      }\
      if (size > 3) {\
        sign_scalar##16x4_t l1 = vld1_##sign_short##16(dat); dat += 4;\
        sum1 = vaddw_##sign_short##16(sum1, l1);\
        size -= 4;\
      }\
      sign_scalar##32x2_t sumd = vadd_##sign_short##32(\
        vget_low_##sign_short##32(sum1), vget_high_##sign_short##32(sum1));\
      sign_scalar##32_t sums = vget_lane_##sign_short##32(sumd, 0) + \
        vget_lane_##sign_short##32(sumd, 1);\
      for (; size > 0; size--) {\
        sign_scalar##32_t l1 = *dat++;\
        sums += l1;\
      }\
      *sum = sums;\
    }\
  } else if (!sum) {\
    for (; size > 15; size -= 16) {\
      sign_scalar##16x8_t l1 = vld1q_##sign_short##16(dat);\
      sign_scalar##16x8_t l2 = vld1q_##sign_short##16(dat + 8); dat += 16;\
      sign_scalar##32x4_t sq1 = vmull_low_##sign_short##16(l1, l1);\
      sign_scalar##32x4_t sq2 = vmull_high_##sign_short##16(l1, l1);\
      sign_scalar##32x4_t sq3 = vmull_low_##sign_short##16(l2, l2);\
      sign_scalar##32x4_t sq4 = vmull_high_##sign_short##16(l2, l2);\
      sumsq1 = vpadalq_##sign_short##32(sumsq1, sq1);\
      sumsq2 = vpadalq_##sign_short##32(sumsq2, sq2);\
      sumsq3 = vpadalq_##sign_short##32(sumsq3, sq3);\
      sumsq4 = vpadalq_##sign_short##32(sumsq4, sq4);\
    }\
    sumsq1 = vaddq_##sign_short##64(sumsq1, sumsq3);\
    sumsq2 = vaddq_##sign_short##64(sumsq2, sumsq4);\
    if (size > 7) {\
      sign_scalar##16x8_t l1 = vld1q_##sign_short##16(dat); dat += 8;\
      sign_scalar##32x4_t sq1 = vmull_low_##sign_short##16(l1, l1);\
      sign_scalar##32x4_t sq2 = vmull_high_##sign_short##16(l1, l1);\
      sumsq1 = vpadalq_##sign_short##32(sumsq1, sq1);\
      sumsq2 = vpadalq_##sign_short##32(sumsq2, sq2);\
      size -= 8;\
    }\
    sumsq1 = vaddq_##sign_short##64(sumsq1, sumsq2);\
    if (size > 3) {\
      sign_scalar##16x4_t l1 = vld1_##sign_short##16(dat); dat += 4;\
      sign_scalar##32x4_t sq1 = vmull_##sign_short##16(l1, l1);\
      sumsq1 = vpadalq_##sign_short##32(sumsq1, sq1);\
      size -= 4;\
    }\
    sign_scalar##64_t sumsqs = vgetq_lane_##sign_short##64(sumsq1, 0) + \
      vgetq_lane_##sign_short##64(sumsq1, 1);\
    for (; size > 0; size--) {\
      sign_scalar##32_t l1 = *dat++;\
      sumsqs += l1 * l1;\
    }\
    *sumsquare = sumsqs;\
  } else {\
    for (; size > 15; size -= 16) {\
      sign_scalar##16x8_t l1 = vld1q_##sign_short##16(dat);\
      sign_scalar##16x8_t l2 = vld1q_##sign_short##16(dat + 8); dat += 16;\
      sum1 = vpadalq_##sign_short##16(sum1, l1);\
      sum2 = vpadalq_##sign_short##16(sum2, l2);\
      sign_scalar##32x4_t sq1 = vmull_low_##sign_short##16(l1, l1);\
      sign_scalar##32x4_t sq2 = vmull_high_##sign_short##16(l1, l1);\
      sign_scalar##32x4_t sq3 = vmull_low_##sign_short##16(l2, l2);\
      sign_scalar##32x4_t sq4 = vmull_high_##sign_short##16(l2, l2);\
      sumsq1 = vpadalq_##sign_short##32(sumsq1, sq1);\
      sumsq2 = vpadalq_##sign_short##32(sumsq2, sq2);\
      sumsq3 = vpadalq_##sign_short##32(sumsq3, sq3);\
      sumsq4 = vpadalq_##sign_short##32(sumsq4, sq4);\
    }\
    sum1 = vaddq_##sign_short##32(sum1, sum2);\
    sumsq1 = vaddq_##sign_short##64(sumsq1, sumsq3);\
    sumsq2 = vaddq_##sign_short##64(sumsq2, sumsq4);\
    if (size > 7) {\
      sign_scalar##16x8_t l1 = vld1q_##sign_short##16(dat); dat += 8;\
      sum1 = vpadalq_##sign_short##16(sum1, l1);\
      sign_scalar##32x4_t sq1 = vmull_low_##sign_short##16(l1, l1);\
      sign_scalar##32x4_t sq2 = vmull_high_##sign_short##16(l1, l1);\
      sumsq1 = vpadalq_##sign_short##32(sumsq1, sq1);\
      sumsq2 = vpadalq_##sign_short##32(sumsq2, sq2);\
      size -= 8;\
    }\
    sumsq1 = vaddq_##sign_short##64(sumsq1, sumsq2);\
    if (size > 3) {\
      sign_scalar##16x4_t l1 = vld1_##sign_short##16(dat); dat += 4;\
      sum1 = vaddw_##sign_short##16(sum1, l1);\
      sign_scalar##32x4_t sq1 = vmull_##sign_short##16(l1, l1);\
      sumsq1 = vpadalq_##sign_short##32(sumsq1, sq1);\
      size -= 4;\
    }\
    sign_scalar##32x2_t sumd = vadd_##sign_short##32(\
      vget_low_##sign_short##32(sum1), vget_high_##sign_short##32(sum1));\
    sign_scalar##32_t sums = vget_lane_##sign_short##32(sumd, 0) + \
      vget_lane_##sign_short##32(sumd, 1);\
    sign_scalar##64_t sumsqs = vgetq_lane_##sign_short##64(sumsq1, 0) + \
      vgetq_lane_##sign_short##64(sumsq1, 1);\
    for (; size > 0; size--) {\
      sign_scalar##32_t l1 = *dat++;\
      sums += l1;\
      sumsqs += l1 * l1;\
    }\
    *sum = sums;\
    *sumsquare = sumsqs;\
  }\
}

#endif

