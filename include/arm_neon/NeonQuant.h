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
 * File:        NeonQuant.h
 * Description: Source code template for NEON quantization kernels.
 *****************************************************************************/

#include "arm_neon/NeonExtreme.h"

#ifndef INCLUDE_NEON_QUANT
#define INCLUDE_NEON_QUANT

static inline void inline_dequant_cvt_f32_s32(
  float *dst, const int32_t *src, float scale, uint32_t size) {

  const float32x4_t sc4 = vdupq_n_f32(scale);
  const float32x2_t sc2 = vdup_n_f32(scale);
  for (; size >= 16; size -= 16) {
    int32x4_t v1 = vld1q_s32(src);
    int32x4_t v2 = vld1q_s32(src + 4);
    int32x4_t v3 = vld1q_s32(src + 8);
    int32x4_t v4 = vld1q_s32(src + 12); src += 16;
    float32x4_t q1 = vcvtq_f32_s32(v1);
    float32x4_t q2 = vcvtq_f32_s32(v2);
    float32x4_t q3 = vcvtq_f32_s32(v3);
    float32x4_t q4 = vcvtq_f32_s32(v4);
    q1 = vmulq_f32(q1, sc4);
    q2 = vmulq_f32(q2, sc4);
    q3 = vmulq_f32(q3, sc4);
    q4 = vmulq_f32(q4, sc4);
    vst1q_f32(dst, q1);
    vst1q_f32(dst + 4, q2);
    vst1q_f32(dst + 8, q3);
    vst1q_f32(dst + 12, q4); dst += 16;
  }
  if (size >= 8) {
    int32x4_t v1 = vld1q_s32(src);
    int32x4_t v2 = vld1q_s32(src + 4); src += 8;
    float32x4_t q1 = vcvtq_f32_s32(v1);
    float32x4_t q2 = vcvtq_f32_s32(v2);
    q1 = vmulq_f32(q1, sc4);
    q2 = vmulq_f32(q2, sc4);
    vst1q_f32(dst, q1);
    vst1q_f32(dst + 4, q2); dst += 8;
    size -= 8;
  }
  if (size >= 4) {
    int32x4_t v1 = vld1q_s32(src); src += 4;
    float32x4_t q1 = vcvtq_f32_s32(v1);
    q1 = vmulq_f32(q1, sc4);
    vst1q_f32(dst, q1); dst += 4;
    size -= 4;
  }
  if (size >= 2) {
    int32x2_t v1 = vld1_s32(src); src += 2;
    float32x2_t d1 = vcvt_f32_s32(v1);
    d1 = vmul_f32(d1, sc2);
    vst1_f32(dst, d1); dst += 2;
    size -= 2;
  }
  if (size >= 1) {
    *dst = (float)(*src) * scale;
  }
}

static inline void inline_quant_asym_u8_from_f32(
  const float32_t *src, uint8_t *dst,
  uint32_t size, uint8_t zero_point, float32_t scale) {

  if (scale <= 0) return;
  if (size == 0) return;
  const float32_t add_zero_s = (float32_t)zero_point + 0.5f;
  const float32x4_t add_zero_q = vdupq_n_f32(add_zero_s);
  const float32_t mult_s = 1.0f / scale;
  const float32x4_t mult_q = vdupq_n_f32(mult_s);

  for (; size >= 16; size -= 16) {
    float32x4_t f1 = vld1q_f32(src);
    float32x4_t f2 = vld1q_f32(src + 4);
    float32x4_t f3 = vld1q_f32(src + 8);
    float32x4_t f4 = vld1q_f32(src + 12); src += 16;
    f1 = vmlaq_f32(add_zero_q, f1, mult_q);
    f2 = vmlaq_f32(add_zero_q, f2, mult_q);
    f3 = vmlaq_f32(add_zero_q, f3, mult_q);
    f4 = vmlaq_f32(add_zero_q, f4, mult_q);
    uint32x4_t u1 = vcvtq_u32_f32(f1);
    uint32x4_t u2 = vcvtq_u32_f32(f2);
    uint32x4_t u3 = vcvtq_u32_f32(f3);
    uint32x4_t u4 = vcvtq_u32_f32(f4);
    uint16x4_t t1 = vqmovn_u32(u1);
    uint16x4_t t2 = vqmovn_u32(u2);
    uint16x4_t t3 = vqmovn_u32(u3);
    uint16x4_t t4 = vqmovn_u32(u4);
    uint8x8_t d1 = vqmovn_u16(vcombine_u16(t1, t2));
    uint8x8_t d2 = vqmovn_u16(vcombine_u16(t3, t4));
    vst1_u8(dst, d1);
    vst1_u8(dst + 8, d2); dst += 16;
  }
  if (size >= 8) {
    float32x4_t f1 = vld1q_f32(src);
    float32x4_t f2 = vld1q_f32(src + 4); src += 8;
    f1 = vmlaq_f32(add_zero_q, f1, mult_q);
    f2 = vmlaq_f32(add_zero_q, f2, mult_q);
    uint32x4_t u1 = vcvtq_u32_f32(f1);
    uint32x4_t u2 = vcvtq_u32_f32(f2);
    uint16x4_t t1 = vqmovn_u32(u1);
    uint16x4_t t2 = vqmovn_u32(u2);
    uint8x8_t d1 = vqmovn_u16(vcombine_u16(t1, t2));
    vst1_u8(dst, d1); dst += 8;
    size -= 8;
  }
  if (size >= 4) {
    float32x4_t f1 = vld1q_f32(src); src += 4;
    f1 = vmlaq_f32(add_zero_q, f1, mult_q);
    uint32x4_t u1 = vcvtq_u32_f32(f1);
    uint16x4_t t1 = vqmovn_u32(u1);
    uint16x4_t z1 = vdup_n_u16(0);
    uint8x8_t d1 = vqmovn_u16(vcombine_u16(t1, z1));
    vst1_lane_u8(dst, d1, 0);
    vst1_lane_u8(dst + 1, d1, 1);
    vst1_lane_u8(dst + 2, d1, 2);
    vst1_lane_u8(dst + 3, d1, 3);
    dst += 4;
    size -= 4;
  }
  for (; size > 0; size--) {
    float32_t f1 = *src++;
    f1 = f1 * mult_s + add_zero_s;
    f1 = f1 < 0 ? 0.0 : f1;
    f1 = f1 > 255 ? 255.0 : f1;
    uint32_t u1 = (uint32_t)f1;
    uint8_t s1 = u1 >= 256 ? 255 : u1;
    *dst = s1; dst++;
  }
}

static inline void inline_quant_asym_u16_from_f32(
  const float32_t *src, uint16_t *dst,
  uint32_t size, uint16_t zero_point, float32_t scale) {

  if (scale <= 0) return;
  if (size == 0) return;
  const float32_t add_zero_s = (float32_t)zero_point + 0.5f;
  const float32x4_t add_zero_q = vdupq_n_f32(add_zero_s);
  const float32_t mult_s = 1.0f / scale;
  const float32x4_t mult_q = vdupq_n_f32(mult_s);

  for (; size >= 16; size -= 16) {
    float32x4_t f1 = vld1q_f32(src);
    float32x4_t f2 = vld1q_f32(src + 4);
    float32x4_t f3 = vld1q_f32(src + 8);
    float32x4_t f4 = vld1q_f32(src + 12); src += 16;
    f1 = vmlaq_f32(add_zero_q, f1, mult_q);
    f2 = vmlaq_f32(add_zero_q, f2, mult_q);
    f3 = vmlaq_f32(add_zero_q, f3, mult_q);
    f4 = vmlaq_f32(add_zero_q, f4, mult_q);
    uint32x4_t u1 = vcvtq_u32_f32(f1);
    uint32x4_t u2 = vcvtq_u32_f32(f2);
    uint32x4_t u3 = vcvtq_u32_f32(f3);
    uint32x4_t u4 = vcvtq_u32_f32(f4);
    uint16x4_t t1 = vqmovn_u32(u1);
    uint16x4_t t2 = vqmovn_u32(u2);
    uint16x4_t t3 = vqmovn_u32(u3);
    uint16x4_t t4 = vqmovn_u32(u4);
    vst1_u16(dst, t1);
    vst1_u16(dst + 4, t2);
    vst1_u16(dst + 8, t3);
    vst1_u16(dst + 12, t4); dst += 16;
  }
  if (size >= 8) {
    float32x4_t f1 = vld1q_f32(src);
    float32x4_t f2 = vld1q_f32(src + 4); src += 8;
    f1 = vmlaq_f32(add_zero_q, f1, mult_q);
    f2 = vmlaq_f32(add_zero_q, f2, mult_q);
    uint32x4_t u1 = vcvtq_u32_f32(f1);
    uint32x4_t u2 = vcvtq_u32_f32(f2);
    uint16x4_t t1 = vqmovn_u32(u1);
    uint16x4_t t2 = vqmovn_u32(u2);
    vst1_u16(dst, t1);
    vst1_u16(dst + 4, t2); dst += 8;
    size -= 8;
  }
  if (size >= 4) {
    float32x4_t f1 = vld1q_f32(src); src += 4;
    f1 = vmlaq_f32(add_zero_q, f1, mult_q);
    uint32x4_t u1 = vcvtq_u32_f32(f1);
    uint16x4_t t1 = vqmovn_u32(u1);
    vst1_u16(dst, t1); dst += 4;
    size -= 4;
  }
  if (size > 0) {
    float32x4_t f1 = vdupq_n_f32(0);
    f1 = vsetq_lane_f32(src[0], f1, 0);
    if (size > 1) f1 = vsetq_lane_f32(src[1], f1, 1);
    if (size > 2) f1 = vsetq_lane_f32(src[2], f1, 2);
    f1 = vmlaq_f32(add_zero_q, f1, mult_q);
    uint32x4_t u1 = vcvtq_u32_f32(f1);
    uint16x4_t t1 = vqmovn_u32(u1);
    vst1_lane_u16(dst, t1, 0);
    if (size > 1) vst1_lane_u16(dst + 1, t1, 1);
    if (size > 2) vst1_lane_u16(dst + 2, t1, 2);
  }
}

#if !__aarch64__
static inline int32x4_t vcvtaq_s32_f32(float32x4_t src) {
  const static float32x4_t cvt_positive_offset = {0.5f, 0.5f, 0.5f, 0.5f};
  const static float32x4_t cvt_negative_offset = {-0.5f, -0.5f, -0.5f, -0.5f};
  const static float32x4_t cmp_ref = {0.0f, 0.0f, 0.0f, 0.0f};
  uint32x4_t mask = vcgtq_f32(src, cmp_ref); //src big, set 1
  float32x4_t offset = vbslq_f32(mask, cvt_positive_offset, cvt_negative_offset);
  src = vaddq_f32(src, offset);
  return vcvtq_s32_f32(src);
}
#endif

static inline void inline_quant_sym_s8_from_f32(
  const float32_t *src, int8_t *dst,
  uint32_t size, float32_t scale) {

  if (scale <= 0) return;
  if (size == 0) return;
  const float32_t mult_s = 1.0f / scale;
  const float32x4_t mult_q = vdupq_n_f32(mult_s);

  for (; size >= 16; size -= 16) {
    float32x4_t f1 = vld1q_f32(src);
    float32x4_t f2 = vld1q_f32(src + 4);
    float32x4_t f3 = vld1q_f32(src + 8);
    float32x4_t f4 = vld1q_f32(src + 12); src += 16;
    f1 = vmulq_f32(f1, mult_q);
    f2 = vmulq_f32(f2, mult_q);
    f3 = vmulq_f32(f3, mult_q);
    f4 = vmulq_f32(f4, mult_q);
    int32x4_t i1 = vcvtaq_s32_f32(f1);
    int32x4_t i2 = vcvtaq_s32_f32(f2);
    int32x4_t i3 = vcvtaq_s32_f32(f3);
    int32x4_t i4 = vcvtaq_s32_f32(f4);
    int16x4_t v1 = vqmovn_s32(i1);
    int16x4_t v2 = vqmovn_s32(i2);
    int16x4_t v3 = vqmovn_s32(i3);
    int16x4_t v4 = vqmovn_s32(i4);
    int8x8_t w1 = vqmovn_s16(vcombine_s16(v1, v2));
    int8x8_t w2 = vqmovn_s16(vcombine_s16(v3, v4));
    vst1_s8(dst, w1);
    vst1_s8(dst + 8, w2); dst += 16;
  }
  if (size >= 8) {
    float32x4_t f1 = vld1q_f32(src);
    float32x4_t f2 = vld1q_f32(src + 4); src += 8;
    f1 = vmulq_f32(f1, mult_q);
    f2 = vmulq_f32(f2, mult_q);
    int32x4_t i1 = vcvtaq_s32_f32(f1);
    int32x4_t i2 = vcvtaq_s32_f32(f2);
    int16x4_t v1 = vqmovn_s32(i1);
    int16x4_t v2 = vqmovn_s32(i2);
    int8x8_t w1 = vqmovn_s16(vcombine_s16(v1, v2));
    vst1_s8(dst, w1); dst += 8;
    size -= 8;
  }
  if (size >= 4) {
    float32x4_t f1 = vld1q_f32(src); src += 4;
    f1 = vmulq_f32(f1, mult_q);
    int32x4_t i1 = vcvtaq_s32_f32(f1);
    int16x4_t v1 = vqmovn_s32(i1);
    int16x4_t z1 = vdup_n_s16(0);
    int8x8_t w1 = vqmovn_s16(vcombine_s16(v1, z1));
    vst1_lane_s8(dst, w1, 0);
    vst1_lane_s8(dst + 1, w1, 1);
    vst1_lane_s8(dst + 2, w1, 2);
    vst1_lane_s8(dst + 3, w1, 3); dst += 4;
    size -= 4;
  }
  for (; size > 0; size--) {
    float32_t f1 = *src++;
    f1 *= mult_s;
    f1 += f1 > 0 ? 0.5f : -0.5f;
    f1 = f1 < -128 ? -128.0 : f1;
    f1 = f1 > 127 ? 127.0 : f1;
    int8_t s1 = f1;
    *dst = s1; dst++;
  }
}

static inline void inline_quant_sym_s16_from_f32(
  const float32_t *src, int16_t *dst,
  uint32_t size, float32_t scale) {

  if (scale <= 0) return;
  if (size == 0) return;
  const float32_t mult_s = 1.0f / scale;
  const float32x4_t mult_q = vdupq_n_f32(mult_s);

  for (; size >= 16; size -= 16) {
    float32x4_t f1 = vld1q_f32(src);
    float32x4_t f2 = vld1q_f32(src + 4);
    float32x4_t f3 = vld1q_f32(src + 8);
    float32x4_t f4 = vld1q_f32(src + 12); src += 16;
    f1 = vmulq_f32(f1, mult_q);
    f2 = vmulq_f32(f2, mult_q);
    f3 = vmulq_f32(f3, mult_q);
    f4 = vmulq_f32(f4, mult_q);
    int32x4_t i1 = vcvtaq_s32_f32(f1);
    int32x4_t i2 = vcvtaq_s32_f32(f2);
    int32x4_t i3 = vcvtaq_s32_f32(f3);
    int32x4_t i4 = vcvtaq_s32_f32(f4);
    int16x4_t v1 = vqmovn_s32(i1);
    int16x4_t v2 = vqmovn_s32(i2);
    int16x4_t v3 = vqmovn_s32(i3);
    int16x4_t v4 = vqmovn_s32(i4);
    vst1_s16(dst, v1);
    vst1_s16(dst + 4, v2);
    vst1_s16(dst + 8, v3);
    vst1_s16(dst + 12, v4); dst += 16;
  }
  if (size >= 8) {
    float32x4_t f1 = vld1q_f32(src);
    float32x4_t f2 = vld1q_f32(src + 4); src += 8;
    f1 = vmulq_f32(f1, mult_q);
    f2 = vmulq_f32(f2, mult_q);
    int32x4_t i1 = vcvtaq_s32_f32(f1);
    int32x4_t i2 = vcvtaq_s32_f32(f2);
    int16x4_t v1 = vqmovn_s32(i1);
    int16x4_t v2 = vqmovn_s32(i2);
    vst1_s16(dst, v1);
    vst1_s16(dst + 4, v2); dst += 8;
    size -= 8;
  }
  if (size >= 4) {
    float32x4_t f1 = vld1q_f32(src); src += 4;
    f1 = vmulq_f32(f1, mult_q);
    int32x4_t i1 = vcvtaq_s32_f32(f1);
    int16x4_t v1 = vqmovn_s32(i1);
    vst1_s16(dst, v1); dst += 4;
    size -= 4;
  }
  if (size > 0) {
    float32x4_t f1 = vdupq_n_f32(0);
    f1 = vsetq_lane_f32(src[0], f1, 0);
    if (size > 1) f1 = vsetq_lane_f32(src[1], f1, 1);
    if (size > 2) f1 = vsetq_lane_f32(src[2], f1, 2);
    f1 = vmulq_f32(f1, mult_q);
    int32x4_t i1 = vcvtaq_s32_f32(f1);
    int16x4_t v1 = vqmovn_s32(i1);
    vst1_lane_s16(dst, v1, 0);
    if (size > 1) vst1_lane_s16(dst + 1, v1, 1);
    if (size > 2) vst1_lane_s16(dst + 2, v1, 2);
  }
}

static inline void inline_requant_asym_u8_from_s32_mulhi(const int32_t *src,
  uint8_t *dst, uint32_t size, uint8_t src_lshift,
  int32_t mult_factor_22redun, uint8_t zero_point) {

  if (size == 0) return;
  const int32x4_t src_sh4 = vdupq_n_s32(src_lshift);
  const int32x4_t mult_v4 = vdupq_n_s32(mult_factor_22redun);
  const int16x4_t add_z4 = vdup_n_s16((int16_t)zero_point << 6);

  for (; size > 15; size -= 16) {
    int32x4_t l1 = vld1q_s32(src);
    int32x4_t l2 = vld1q_s32(src + 4);
    int32x4_t l3 = vld1q_s32(src + 8);
    int32x4_t l4 = vld1q_s32(src + 12); src += 16;
    l1 = vqrshlq_s32(l1, src_sh4);
    l2 = vqrshlq_s32(l2, src_sh4);
    l3 = vqrshlq_s32(l3, src_sh4);
    l4 = vqrshlq_s32(l4, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l2 = vqrdmulhq_s32(l2, mult_v4);
    l3 = vqrdmulhq_s32(l3, mult_v4);
    l4 = vqrdmulhq_s32(l4, mult_v4);
    int16x4_t m1 = vrshrn_n_s32(l1, 16);
    int16x4_t m2 = vrshrn_n_s32(l2, 16);
    int16x4_t m3 = vrshrn_n_s32(l3, 16);
    int16x4_t m4 = vrshrn_n_s32(l4, 16);
    m1 = vadd_s16(m1, add_z4);
    m2 = vadd_s16(m2, add_z4);
    m3 = vadd_s16(m3, add_z4);
    m4 = vadd_s16(m4, add_z4);
    uint8x8_t u1 = vqrshrun_n_s16(vcombine_s16(m1, m2), 6);
    uint8x8_t u2 = vqrshrun_n_s16(vcombine_s16(m3, m4), 6);
    vst1_u8(dst, u1);
    vst1_u8(dst + 8, u2); dst += 16;
  }
  if (size > 7) {
    int32x4_t l1 = vld1q_s32(src);
    int32x4_t l2 = vld1q_s32(src + 4); src += 8;
    l1 = vqrshlq_s32(l1, src_sh4);
    l2 = vqrshlq_s32(l2, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l2 = vqrdmulhq_s32(l2, mult_v4);
    int16x4_t m1 = vrshrn_n_s32(l1, 16);
    int16x4_t m2 = vrshrn_n_s32(l2, 16);
    m1 = vadd_s16(m1, add_z4);
    m2 = vadd_s16(m2, add_z4);
    uint8x8_t u1 = vqrshrun_n_s16(vcombine_s16(m1, m2), 6);
    vst1_u8(dst, u1); dst += 8;
    size -= 8;
  }
  if (size > 3) {
    int32x4_t l1 = vld1q_s32(src); src += 4;
    l1 = vqrshlq_s32(l1, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    int16x4_t m1 = vrshrn_n_s32(l1, 16);
    m1 = vadd_s16(m1, add_z4);
    uint8x8_t u1 = vqrshrun_n_s16(vcombine_s16(m1, m1), 6);
    vst1_lane_u8(dst, u1, 0);
    vst1_lane_u8(dst + 1, u1, 1);
    vst1_lane_u8(dst + 2, u1, 2);
    vst1_lane_u8(dst + 3, u1, 3); dst += 4;
    size -= 4;
  }
  if (size > 0) {
    int32x4_t l1 = vdupq_n_s32(0);
    l1 = vsetq_lane_s32(src[0], l1, 0);
    if (size > 1) l1 = vsetq_lane_s32(src[1], l1, 1);
    if (size > 2) l1 = vsetq_lane_s32(src[2], l1, 2);
    l1 = vqrshlq_s32(l1, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    int16x4_t m1 = vrshrn_n_s32(l1, 16);
    m1 = vadd_s16(m1, add_z4);
    uint8x8_t u1 = vqrshrun_n_s16(vcombine_s16(m1, m1), 6);
    vst1_lane_u8(dst, u1, 0);
    if (size > 1) vst1_lane_u8(dst + 1, u1, 1);
    if (size > 2) vst1_lane_u8(dst + 2, u1, 2);
  }
}

static inline void inline_requant_sym_s8_from_s32_mulhi(const int32_t *src,
  int8_t *dst, uint32_t size,
  uint8_t src_lshift, int32_t mult_factor_22redun) {

  if (size == 0) return;
  const int32x4_t src_sh4 = vdupq_n_s32(src_lshift);
  const int32x4_t mult_v4 = vdupq_n_s32(mult_factor_22redun);

  for (; size > 15; size -= 16) {
    int32x4_t l1 = vld1q_s32(src);
    int32x4_t l2 = vld1q_s32(src + 4);
    int32x4_t l3 = vld1q_s32(src + 8);
    int32x4_t l4 = vld1q_s32(src + 12); src += 16;
    l1 = vqrshlq_s32(l1, src_sh4);
    l2 = vqrshlq_s32(l2, src_sh4);
    l3 = vqrshlq_s32(l3, src_sh4);
    l4 = vqrshlq_s32(l4, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l2 = vqrdmulhq_s32(l2, mult_v4);
    l3 = vqrdmulhq_s32(l3, mult_v4);
    l4 = vqrdmulhq_s32(l4, mult_v4);
    int16x4_t m1 = vrshrn_n_s32(l1, 16);
    int16x4_t m2 = vrshrn_n_s32(l2, 16);
    int16x4_t m3 = vrshrn_n_s32(l3, 16);
    int16x4_t m4 = vrshrn_n_s32(l4, 16);
    int8x8_t s1 = vqrshrn_n_s16(vcombine_s16(m1, m2), 7);
    int8x8_t s2 = vqrshrn_n_s16(vcombine_s16(m3, m4), 7);
    vst1_s8(dst, s1);
    vst1_s8(dst + 8, s2); dst += 16;
  }
  if (size > 7) {
    int32x4_t l1 = vld1q_s32(src);
    int32x4_t l2 = vld1q_s32(src + 4); src += 8;
    l1 = vqrshlq_s32(l1, src_sh4);
    l2 = vqrshlq_s32(l2, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l2 = vqrdmulhq_s32(l2, mult_v4);
    int16x4_t m1 = vrshrn_n_s32(l1, 16);
    int16x4_t m2 = vrshrn_n_s32(l2, 16);
    int8x8_t s1 = vqrshrn_n_s16(vcombine_s16(m1, m2), 7);
    vst1_s8(dst, s1); dst += 8;
    size -= 8;
  }
  if (size > 3) {
    int32x4_t l1 = vld1q_s32(src); src += 4;
    l1 = vqrshlq_s32(l1, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    int16x4_t m1 = vrshrn_n_s32(l1, 16);
    int8x8_t s1 = vqrshrn_n_s16(vcombine_s16(m1, m1), 7);
    vst1_lane_s8(dst, s1, 0);
    vst1_lane_s8(dst + 1, s1, 1);
    vst1_lane_s8(dst + 2, s1, 2);
    vst1_lane_s8(dst + 3, s1, 3); dst += 4;
    size -= 4;
  }
  if (size > 0) {
    int32x4_t l1 = vdupq_n_s32(0);
    l1 = vsetq_lane_s32(src[0], l1, 0);
    if (size > 1) l1 = vsetq_lane_s32(src[1], l1, 1);
    if (size > 2) l1 = vsetq_lane_s32(src[2], l1, 2);
    l1 = vqrshlq_s32(l1, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    int16x4_t m1 = vrshrn_n_s32(l1, 16);
    int8x8_t s1 = vqrshrn_n_s16(vcombine_s16(m1, m1), 7);
    vst1_lane_s8(dst, s1, 0);
    if (size > 1) vst1_lane_s8(dst + 1, s1, 1);
    if (size > 2) vst1_lane_s8(dst + 2, s1, 2);
  }
}

static inline void inline_requant_asym_u16_from_s32_mulhi(const int32_t *src,
  uint16_t *dst, uint32_t size, uint8_t src_lshift,
  int32_t mult_factor, uint16_t zero_point) {

  if (size == 0) return;
  const int32x4_t src_sh4 = vdupq_n_s32(src_lshift);
  const int32x4_t mult_v4 = vdupq_n_s32(mult_factor);
  const int32x4_t add_z4 = vdupq_n_s32((int32_t)zero_point << 14);

  for (; size > 15; size -= 16) {
    int32x4_t l1 = vld1q_s32(src);
    int32x4_t l2 = vld1q_s32(src + 4);
    int32x4_t l3 = vld1q_s32(src + 8);
    int32x4_t l4 = vld1q_s32(src + 12); src += 16;
    l1 = vqrshlq_s32(l1, src_sh4);
    l2 = vqrshlq_s32(l2, src_sh4);
    l3 = vqrshlq_s32(l3, src_sh4);
    l4 = vqrshlq_s32(l4, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l2 = vqrdmulhq_s32(l2, mult_v4);
    l3 = vqrdmulhq_s32(l3, mult_v4);
    l4 = vqrdmulhq_s32(l4, mult_v4);
    l1 = vqaddq_s32(l1, add_z4);
    l2 = vqaddq_s32(l2, add_z4);
    l3 = vqaddq_s32(l3, add_z4);
    l4 = vqaddq_s32(l4, add_z4);
    uint16x4_t m1 = vqrshrun_n_s32(l1, 14);
    uint16x4_t m2 = vqrshrun_n_s32(l2, 14);
    uint16x4_t m3 = vqrshrun_n_s32(l3, 14);
    uint16x4_t m4 = vqrshrun_n_s32(l4, 14);
    vst1_u16(dst, m1);
    vst1_u16(dst + 4, m2);
    vst1_u16(dst + 8, m3);
    vst1_u16(dst + 12, m4); dst += 16;
  }
  for (; size > 7; size -= 8) {
    int32x4_t l1 = vld1q_s32(src);
    int32x4_t l2 = vld1q_s32(src + 4); src += 8;
    l1 = vqrshlq_s32(l1, src_sh4);
    l2 = vqrshlq_s32(l2, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l2 = vqrdmulhq_s32(l2, mult_v4);
    l1 = vqaddq_s32(l1, add_z4);
    l2 = vqaddq_s32(l2, add_z4);
    uint16x4_t m1 = vqrshrun_n_s32(l1, 14);
    uint16x4_t m2 = vqrshrun_n_s32(l2, 14);
    vst1_u16(dst, m1);
    vst1_u16(dst + 4, m2); dst += 8;
  }
  for (; size > 3; size -= 4) {
    int32x4_t l1 = vld1q_s32(src); src += 4;
    l1 = vqrshlq_s32(l1, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l1 = vqaddq_s32(l1, add_z4);
    uint16x4_t m1 = vqrshrun_n_s32(l1, 14);
    vst1_u16(dst, m1); dst += 4;
  }
  if (size > 0) {
    int32x4_t l1 = vdupq_n_s32(0);
    l1 = vsetq_lane_s32(src[0], l1, 0);
    if (size > 1) l1 = vsetq_lane_s32(src[1], l1, 1);
    if (size > 2) l1 = vsetq_lane_s32(src[2], l1, 2);
    l1 = vqrshlq_s32(l1, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l1 = vqaddq_s32(l1, add_z4);
    uint16x4_t m1 = vqrshrun_n_s32(l1, 14);
    vst1_lane_u16(dst, m1, 0);
    if (size > 1) vst1_lane_u16(dst + 1, m1, 1);
    if (size > 2) vst1_lane_u16(dst + 2, m1, 2);
  }
}

static inline void inline_requant_sym_s16_from_s32_mulhi(const int32_t *src,
  int16_t *dst, uint32_t size,
  uint8_t src_lshift, int32_t mult_factor) {

  if (size == 0) return;
  const int32x4_t src_sh4 = vdupq_n_s32(src_lshift);
  const int32x4_t mult_v4 = vdupq_n_s32(mult_factor);

  for (; size > 15; size -= 16) {
    int32x4_t l1 = vld1q_s32(src);
    int32x4_t l2 = vld1q_s32(src + 4);
    int32x4_t l3 = vld1q_s32(src + 8);
    int32x4_t l4 = vld1q_s32(src + 12); src += 16;
    l1 = vqrshlq_s32(l1, src_sh4);
    l2 = vqrshlq_s32(l2, src_sh4);
    l3 = vqrshlq_s32(l3, src_sh4);
    l4 = vqrshlq_s32(l4, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l2 = vqrdmulhq_s32(l2, mult_v4);
    l3 = vqrdmulhq_s32(l3, mult_v4);
    l4 = vqrdmulhq_s32(l4, mult_v4);
    int16x4_t m1 = vqrshrn_n_s32(l1, 15);
    int16x4_t m2 = vqrshrn_n_s32(l2, 15);
    int16x4_t m3 = vqrshrn_n_s32(l3, 15);
    int16x4_t m4 = vqrshrn_n_s32(l4, 15);
    vst1_s16(dst, m1);
    vst1_s16(dst + 4, m2);
    vst1_s16(dst + 8, m3);
    vst1_s16(dst + 12, m4); dst += 16;
  }
  if (size > 7) {
    int32x4_t l1 = vld1q_s32(src);
    int32x4_t l2 = vld1q_s32(src + 4); src += 8;
    l1 = vqrshlq_s32(l1, src_sh4);
    l2 = vqrshlq_s32(l2, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    l2 = vqrdmulhq_s32(l2, mult_v4);
    int16x4_t m1 = vqrshrn_n_s32(l1, 15);
    int16x4_t m2 = vqrshrn_n_s32(l2, 15);
    vst1_s16(dst, m1);
    vst1_s16(dst + 4, m2); dst += 8;
    size -= 8;
  }
  if (size > 3) {
    int32x4_t l1 = vld1q_s32(src); src += 4;
    l1 = vqrshlq_s32(l1, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    int16x4_t m1 = vqrshrn_n_s32(l1, 15);
    vst1_s16(dst, m1); dst += 4;
    size -= 4;
  }
  if (size > 0) {
    int32x4_t l1 = vdupq_n_s32(0);
    l1 = vsetq_lane_s32(src[0], l1, 0);
    if (size > 1) l1 = vsetq_lane_s32(src[1], l1, 1);
    if (size > 2) l1 = vsetq_lane_s32(src[2], l1, 2);
    l1 = vqrshlq_s32(l1, src_sh4);
    l1 = vqrdmulhq_s32(l1, mult_v4);
    int16x4_t m1 = vqrshrn_n_s32(l1, 15);
    vst1_lane_s16(dst, m1, 0);
    if (size > 1) vst1_lane_s16(dst + 1, m1, 1);
    if (size > 2) vst1_lane_s16(dst + 2, m1, 2);
  }
}

static inline void inline_requant_asym_u8_from_s16_mulhi(const int16_t *src,
  uint8_t *dst, uint32_t size, uint8_t src_lshift,
  int16_t mult_factor, uint8_t zero_point) {

  if (size == 0) return;
  const int16x8_t src_sh8 = vdupq_n_s16(src_lshift);
  const int16x8_t mult_v8 = vdupq_n_s16(mult_factor);
  const int16x8_t add_z8 = vdupq_n_s16((int16_t)zero_point << 6);

  for (; size > 31; size -= 32) {
    int16x8_t l1 = vld1q_s16(src);
    int16x8_t l2 = vld1q_s16(src + 8);
    int16x8_t l3 = vld1q_s16(src + 16);
    int16x8_t l4 = vld1q_s16(src + 24); src += 32;
    l1 = vqrshlq_s16(l1, src_sh8);
    l2 = vqrshlq_s16(l2, src_sh8);
    l3 = vqrshlq_s16(l3, src_sh8);
    l4 = vqrshlq_s16(l4, src_sh8);
    l1 = vqrdmulhq_s16(l1, mult_v8);
    l2 = vqrdmulhq_s16(l2, mult_v8);
    l3 = vqrdmulhq_s16(l3, mult_v8);
    l4 = vqrdmulhq_s16(l4, mult_v8);
    l1 = vqaddq_s16(l1, add_z8);
    l2 = vqaddq_s16(l2, add_z8);
    l3 = vqaddq_s16(l3, add_z8);
    l4 = vqaddq_s16(l4, add_z8);
    uint8x8_t m1 = vqrshrun_n_s16(l1, 6);
    uint8x8_t m2 = vqrshrun_n_s16(l2, 6);
    uint8x8_t m3 = vqrshrun_n_s16(l3, 6);
    uint8x8_t m4 = vqrshrun_n_s16(l4, 6);
    vst1_u8(dst, m1);
    vst1_u8(dst + 8, m2);
    vst1_u8(dst + 16, m3);
    vst1_u8(dst + 24, m4); dst += 32;
  }
  if (size > 15) {
    int16x8_t l1 = vld1q_s16(src);
    int16x8_t l2 = vld1q_s16(src + 8); src += 16;
    l1 = vqrshlq_s16(l1, src_sh8);
    l2 = vqrshlq_s16(l2, src_sh8);
    l1 = vqrdmulhq_s16(l1, mult_v8);
    l2 = vqrdmulhq_s16(l2, mult_v8);
    l1 = vqaddq_s16(l1, add_z8);
    l2 = vqaddq_s16(l2, add_z8);
    uint8x8_t m1 = vqrshrun_n_s16(l1, 6);
    uint8x8_t m2 = vqrshrun_n_s16(l2, 6);
    vst1_u8(dst, m1);
    vst1_u8(dst + 8, m2); dst += 16;
    size -= 16;
  }
  if (size > 7) {
    int16x8_t l1 = vld1q_s16(src); src += 8;
    l1 = vqrshlq_s16(l1, src_sh8);
    l1 = vqrdmulhq_s16(l1, mult_v8);
    l1 = vqaddq_s16(l1, add_z8);
    uint8x8_t m1 = vqrshrun_n_s16(l1, 6);
    vst1_u8(dst, m1); dst += 8;
    size -= 8;
  }
  if (size > 3) {
    int16x4_t l1 = vld1_s16(src); src += 4;
    l1 = vqrshl_s16(l1, vget_low_s16(src_sh8));
    l1 = vqrdmulh_s16(l1, vget_low_s16(mult_v8));
    l1 = vqadd_s16(l1, vget_low_s16(add_z8));
    uint8x8_t m1 = vqrshrun_n_s16(vcombine_s16(l1, vdup_n_s16(0)), 6);
    vst1_lane_u8(dst, m1, 0);
    vst1_lane_u8(dst + 1, m1, 1);
    vst1_lane_u8(dst + 2, m1, 2);
    vst1_lane_u8(dst + 3, m1, 3); dst += 4;
    size -= 4;
  }
  if (size > 0) {
    int16x4_t l1 = vdup_n_s16(0);
    l1 = vset_lane_s16(src[0], l1, 0);
    if (size > 1) l1 = vset_lane_s16(src[1], l1, 1);
    if (size > 2) l1 = vset_lane_s16(src[2], l1, 2);
    l1 = vqrshl_s16(l1, vget_low_s16(src_sh8));
    l1 = vqrdmulh_s16(l1, vget_low_s16(mult_v8));
    l1 = vqadd_s16(l1, vget_low_s16(add_z8));
    uint8x8_t m1 = vqrshrun_n_s16(vcombine_s16(l1, vdup_n_s16(0)), 6);
    vst1_lane_u8(dst, m1, 0);
    if (size > 1) vst1_lane_u8(dst + 1, m1, 1);
    if (size > 2) vst1_lane_u8(dst + 2, m1, 2);
  }
}

static inline void inline_requant_sym_s8_from_s16_mulhi(const int16_t *src,
  int8_t *dst, uint32_t size,
  uint8_t src_lshift, int16_t mult_factor) {

  if (size == 0) return;
  const int16x8_t src_sh8 = vdupq_n_s16(src_lshift);
  const int16x8_t mult_v8 = vdupq_n_s16(mult_factor);

  for (; size > 31; size -= 32) {
    int16x8_t l1 = vld1q_s16(src);
    int16x8_t l2 = vld1q_s16(src + 8);
    int16x8_t l3 = vld1q_s16(src + 16);
    int16x8_t l4 = vld1q_s16(src + 24); src += 32;
    l1 = vqrshlq_s16(l1, src_sh8);
    l2 = vqrshlq_s16(l2, src_sh8);
    l3 = vqrshlq_s16(l3, src_sh8);
    l4 = vqrshlq_s16(l4, src_sh8);
    l1 = vqrdmulhq_s16(l1, mult_v8);
    l2 = vqrdmulhq_s16(l2, mult_v8);
    l3 = vqrdmulhq_s16(l3, mult_v8);
    l4 = vqrdmulhq_s16(l4, mult_v8);
    int8x8_t m1 = vqrshrn_n_s16(l1, 7);
    int8x8_t m2 = vqrshrn_n_s16(l2, 7);
    int8x8_t m3 = vqrshrn_n_s16(l3, 7);
    int8x8_t m4 = vqrshrn_n_s16(l4, 7);
    vst1_s8(dst, m1);
    vst1_s8(dst + 8, m2);
    vst1_s8(dst + 16, m3);
    vst1_s8(dst + 24, m4); dst += 32;
  }
  if (size > 15) {
    int16x8_t l1 = vld1q_s16(src);
    int16x8_t l2 = vld1q_s16(src + 8); src += 16;
    l1 = vqrshlq_s16(l1, src_sh8);
    l2 = vqrshlq_s16(l2, src_sh8);
    l1 = vqrdmulhq_s16(l1, mult_v8);
    l2 = vqrdmulhq_s16(l2, mult_v8);
    int8x8_t m1 = vqrshrn_n_s16(l1, 7);
    int8x8_t m2 = vqrshrn_n_s16(l2, 7);
    vst1_s8(dst, m1);
    vst1_s8(dst + 8, m2); dst += 16;
    size -= 16;
  }
  if (size > 7) {
    int16x8_t l1 = vld1q_s16(src); src += 8;
    l1 = vqrshlq_s16(l1, src_sh8);
    l1 = vqrdmulhq_s16(l1, mult_v8);
    int8x8_t m1 = vqrshrn_n_s16(l1, 7);
    vst1_s8(dst, m1); dst += 8;
    size -= 8;
  }
  if (size > 3) {
    int16x4_t l1 = vld1_s16(src); src += 4;
    l1 = vqrshl_s16(l1, vget_low_s16(src_sh8));
    l1 = vqrdmulh_s16(l1, vget_low_s16(mult_v8));
    int8x8_t m1 = vqrshrn_n_s16(vcombine_s16(l1, vdup_n_s16(0)), 7);
    vst1_lane_s8(dst, m1, 0);
    vst1_lane_s8(dst + 1, m1, 1);
    vst1_lane_s8(dst + 2, m1, 2);
    vst1_lane_s8(dst + 3, m1, 3); dst += 4;
    size -= 4;
  }
  if (size > 0) {
    int16x4_t l1 = vdup_n_s16(0);
    l1 = vset_lane_s16(src[0], l1, 0);
    if (size > 1) l1 = vset_lane_s16(src[1], l1, 1);
    if (size > 2) l1 = vset_lane_s16(src[2], l1, 2);
    l1 = vqrshl_s16(l1, vget_low_s16(src_sh8));
    l1 = vqrdmulh_s16(l1, vget_low_s16(mult_v8));
    int8x8_t m1 = vqrshrn_n_s16(vcombine_s16(l1, vdup_n_s16(0)), 7);
    vst1_lane_s8(dst, m1, 0);
    if (size > 1) vst1_lane_s8(dst + 1, m1, 1);
    if (size > 2) vst1_lane_s8(dst + 2, m1, 2);
  }
}

#endif
