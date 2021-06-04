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
 * File:        NeonIntOpSign.h
 * Description: Sign-irrelevant representations of NEON intrinsics and
 *              ASMs for integer operations involved in 8->32bit GEMM.
 *              With a macro as (signed/unsigned) switch, these
 *              representations are converted to the corresponding
 *              type of intrinsics/ASMs.
 *****************************************************************************/

#include <stdint.h>
#include <arm_neon.h>

#ifndef INCLUDE_NEON_INTEGER_SIGN
#define INCLUDE_NEON_INTEGER_SIGN

#ifdef GEMM_UNSIGNED_INT
#define I8 uint8_t
#define I16 uint16_t
#define I32 uint32_t
#define I8X8 uint8x8_t
#define I8X16 uint8x16_t
#define I16X4 uint16x4_t
#define I16X8 uint16x8_t
#define I32X2 uint32x2_t
#define I32X4 uint32x4_t
#define I64X2 uint64x2_t
#define I8X8X2 uint8x8x2_t
#define I8X16X2 uint8x16x2_t
#define I16X4X2 uint16x4x2_t
#define I16X8X2 uint16x8x2_t
#define I32X2X2 uint32x2x2_t
#define I32X4X2 uint32x4x2_t
#define I8X8X3 uint8x8x3_t
#define I8X16X3 uint8x16x3_t
#define I16X4X3 uint16x4x3_t
#define I16X8X3 uint16x8x3_t
#define I32X2X3 uint32x2x3_t
#define I32X4X3 uint32x4x3_t
#define I8X8X4 uint8x8x4_t
#define I8X16X4 uint8x16x4_t
#define I16X4X4 uint16x4x4_t
#define I16X8X4 uint16x8x4_t
#define I32X2X4 uint32x2x4_t
#define I32X4X4 uint32x4x4_t
#else
#define I8 int8_t
#define I16 int16_t
#define I32 int32_t
#define I8X8 int8x8_t
#define I8X16 int8x16_t
#define I16X4 int16x4_t
#define I16X8 int16x8_t
#define I32X2 int32x2_t
#define I32X4 int32x4_t
#define I64X2 int64x2_t
#define I8X8X2 int8x8x2_t
#define I8X16X2 int8x16x2_t
#define I16X4X2 int16x4x2_t
#define I16X8X2 int16x8x2_t
#define I32X2X2 int32x2x2_t
#define I32X4X2 int32x4x2_t
#define I8X8X3 int8x8x3_t
#define I8X16X3 int8x16x3_t
#define I16X4X3 int16x4x3_t
#define I16X8X3 int16x8x3_t
#define I32X2X3 int32x2x3_t
#define I32X4X3 int32x4x3_t
#define I8X8X4 int8x8x4_t
#define I8X16X4 int8x16x4_t
#define I16X4X4 int16x4x4_t
#define I16X8X4 int16x8x4_t
#define I32X2X4 int32x2x4_t
#define I32X4X4 int32x4x4_t
#endif

/* asm instruction switch */
#if __aarch64__
#ifdef GEMM_UNSIGNED_INT
#define IMLAL "umlal"
#define IMLAL2 "umlal2"
#define ISHLL "ushll"
#define ISHLL2 "ushll2"
#define IXTL "uxtl"
#define IADALP "uadalp"
#define IMULL "umull"
#define IMULL2 "umull2"
#define IDOT "udot"
#else
#define IMLAL "smlal"
#define IMLAL2 "smlal2"
#define ISHLL "sshll"
#define ISHLL2 "sshll2"
#define IXTL "sxtl"
#define IADALP "sadalp"
#define IMULL "smull"
#define IMULL2 "smull2"
#define IDOT "sdot"
#endif
#else //armv7a
#ifdef GEMM_UNSIGNED_INT
#define ASM_VMLAL_I16 "vmlal.u16"
#define ASM_VMOVL_I8 "vmovl.u8"
#define ASM_VPADAL_I16 "vpadal.u16"
#define ASM_VMULL_I8 "vmull.u8"
#else
#define ASM_VMLAL_I16 "vmlal.s16"
#define ASM_VMOVL_I8 "vmovl.s8"
#define ASM_VPADAL_I16 "vpadal.s16"
#define ASM_VMULL_I8 "vmull.s8"
#endif
#endif

/* intrinsic function switch */
#ifdef GEMM_UNSIGNED_INT
#define VMLAQ_N_I32(a, b, c) vmlaq_n_u32(a, b, c)
#define VMLA_N_I32(a, b, c) vmla_n_u32(a, b, c)
#define VMLA_I32(a, b, c) vmla_u32(a, b, c)
#define VMLA_LANE_I32(a, b, c, d) vmla_lane_u32(a, b, c, d)
#define VLD1_I8(a) vld1_u8(a)
#define VLD1Q_I8(a) vld1q_u8(a)
#define VLD1_I16(a) vld1_u16(a)
#define VLD1Q_I16(a) vld1q_u16(a)
#define VLD1_I32(a) vld1_u32(a)
#define VLD1Q_I32(a) vld1q_u32(a)
#define VLD3Q_I32(a) vld3q_u32(a)
#define VMOVL_I8(a) vmovl_u8(a)
#define VMOVL_HIGH_I8(a) vmovl_high_u8(a)
#define VST1_I32(a, b) vst1_u32(a, b)
#define VST2_I32(a, b) vst2_u32(a, b)
#define VST3_I32(a, b) vst3_u32(a, b)
#define VST4_I32(a, b) vst4_u32(a, b)
#define VST1Q_I32(a, b) vst1q_u32(a, b)
#define VST2Q_I32(a, b) vst2q_u32(a, b)
#define VST3Q_I32(a, b) vst3q_u32(a, b)
#define VST4Q_I32(a, b) vst4q_u32(a, b)
#define VST1_LANE_I32(a, b, c) vst1_lane_u32(a, b, c)
#define VST1Q_LANE_I32(a, b, c) vst1q_lane_u32(a, b, c)
#define VST2_LANE_I32(a, b, c) vst2_lane_u32(a, b, c)
#define VST2Q_LANE_I32(a, b, c) vst2q_lane_u32(a, b, c)
#define VST3_LANE_I32(a, b, c) vst3_lane_u32(a, b, c)
#define VST3Q_LANE_I32(a, b, c) vst3q_lane_u32(a, b, c)
#define VST4_LANE_I32(a, b, c) vst4_lane_u32(a, b, c)
#define VST4Q_LANE_I32(a, b, c) vst4q_lane_u32(a, b, c)
#define VST1_I16(a, b) vst1_u16(a, b)
#define VST1Q_I16(a, b) vst1q_u16(a, b)
#define VST1_LANE_I16(a, b, c) vst1_lane_u16(a, b, c)
#define VST1Q_LANE_I16(a, b, c) vst1q_lane_u16(a, b, c)
#define VST2_LANE_I16(a, b, c) vst2_lane_u16(a, b, c)
#define VST2Q_LANE_I16(a, b, c) vst2q_lane_u16(a, b, c)
#define VST3_LANE_I16(a, b, c) vst3_lane_u16(a, b, c)
#define VST3Q_LANE_I16(a, b, c) vst3q_lane_u16(a, b, c)
#define VST4_LANE_I16(a, b, c) vst4_lane_u16(a, b, c)
#define VST4Q_LANE_I16(a, b, c) vst4q_lane_u16(a, b, c)
#define VMLAL_LANE_I16(a, b, c, d) vmlal_lane_u16(a, b, c, d)
#define VMLAL_HIGH_LANE_I16(a, b, c, d) vmlal_high_lane_u16(a, b, c, d)
#define VMLAL_N_I16(a, b, c) vmlal_n_u16(a, b, c)
#define VMLAL_HIGH_N_I16(a, b, c) vmlal_high_n_u16(a, b, c)
#define VMLAL_I16(a, b, c) vmlal_u16(a, b, c)
#define VPADAL_I16(a, b) vpadal_u16(a, b)
#define VPADALQ_I16(a, b) vpadalq_u16(a, b)
#define VPADD_I32(a, b) vpadd_u32(a, b)
#define VMULL_I8(a, b) vmull_u8(a, b)
#define VMULL_HIGH_I8(a, b) vmull_high_u8(a, b)
#define VGET_LOW_I8(a) vget_low_u8(a)
#define VGET_HIGH_I8(a) vget_high_u8(a)
#define VGET_LOW_I16(a) vget_low_u16(a)
#define VGET_HIGH_I16(a) vget_high_u16(a)
#define VGET_LANE_I16(a, b) vget_lane_u16(a, b)
#define VGETQ_LANE_I16(a, b) vgetq_lane_u16(a, b)
#define VGET_LOW_I32(a) vget_low_u32(a)
#define VGET_HIGH_I32(a) vget_high_u32(a)
#define VGET_LANE_I32(a, b) vget_lane_u32(a, b)
#define VGETQ_LANE_I32(a, b) vgetq_lane_u32(a, b)
#define VDUP_N_I32(a) vdup_n_u32(a)
#define VDUPQ_N_I32(a) vdupq_n_u32(a)
#define VDUP_N_I16(a) vdup_n_u16(a)
#define VDUPQ_N_I16(a) vdupq_n_u16(a)
#define VDUP_N_I8(a) vdup_n_u8(a)
#define VDUPQ_N_I8(a) vdupq_n_u8(a)
#define VSET_LANE_I32(a, b, c) vset_lane_u32(a, b, c)
#define VSETQ_LANE_I32(a, b, c) vsetq_lane_u32(a, b, c)
#define VSET_LANE_I16(a, b, c) vset_lane_u16(a, b, c)
#define VSETQ_LANE_I16(a, b, c) vsetq_lane_u16(a, b, c)
#define VZIP_I16(a, b) vzip_u16(a, b)
#define VUZP_I16(a, b) vuzp_u16(a, b)
#define VZIPQ_I32(a, b) vzipq_u32(a, b)
#define VZIP1_I32(a, b) vzip1_u32(a, b)
#define VZIP2_I32(a, b) vzip2_u32(a, b)
#define VZIP1Q_I32(a, b) vzip1q_u32(a, b)
#define VZIP2Q_I32(a, b) vzip2q_u32(a, b)
#define VZIP1Q_I64(a, b) vzip1q_u64(a, b)
#define VZIP2Q_I64(a, b) vzip2q_u64(a, b)
#define VTRN_I16(a, b) vtrn_u16(a, b)
#define VADD_I32(a, b) vadd_u32(a, b)
#define VADDQ_I32(a, b) vaddq_u32(a, b)
#define VCOMBINE_I32(a, b) vcombine_u32(a, b)
#define VDOT_I32(a, b, c) vdot_u32(a, b, c)
#define VDOT_LANE_I32(a, b, c, d) vdot_lane_u32(a, b, c, d)
#define VDOTQ_I32(a, b, c) vdotq_u32(a, b, c)
#define VDOTQ_LANE_I32(a, b, c, d) vdotq_lane_u32(a, b, c, d)
#define VDOTQ_LANEQ_I32(a, b, c, d) vdotq_laneq_u32(a, b, c, d)
#define VREINTERPRETQ_I32_I64(a) vreinterpretq_u32_u64(a)
#define VREINTERPRETQ_I64_I32(a) vreinterpretq_u64_u32(a)
#define VREINTERPRET_I8_I32(a) vreinterpret_u8_u32(a)
#define VREINTERPRETQ_I8_I32(a) vreinterpretq_u8_u32(a)
#define VREINTERPRET_I32_I8(a) vreinterpret_u32_u8(a)
#define VREINTERPRETQ_I32_I8(a) vreinterpretq_u32_u8(a)
#else
#define VMLAQ_N_I32(a, b, c) vmlaq_n_s32(a, b, c)
#define VMLA_N_I32(a, b, c) vmla_n_s32(a, b, c)
#define VMLA_I32(a, b, c) vmla_s32(a, b, c)
#define VMLA_LANE_I32(a, b, c, d) vmla_lane_s32(a, b, c, d)
#define VLD1_I8(a) vld1_s8(a)
#define VLD1Q_I8(a) vld1q_s8(a)
#define VLD1_I16(a) vld1_s16(a)
#define VLD1Q_I16(a) vld1q_s16(a)
#define VLD1_I32(a) vld1_s32(a)
#define VLD1Q_I32(a) vld1q_s32(a)
#define VLD3Q_I32(a) vld3q_s32(a)
#define VMOVL_I8(a) vmovl_s8(a)
#define VMOVL_HIGH_I8(a) vmovl_high_s8(a)
#define VST1_I32(a, b) vst1_s32(a, b)
#define VST2_I32(a, b) vst2_s32(a, b)
#define VST3_I32(a, b) vst3_s32(a, b)
#define VST4_I32(a, b) vst4_s32(a, b)
#define VST1Q_I32(a, b) vst1q_s32(a, b)
#define VST2Q_I32(a, b) vst2q_s32(a, b)
#define VST3Q_I32(a, b) vst3q_s32(a, b)
#define VST4Q_I32(a, b) vst4q_s32(a, b)
#define VST1_LANE_I32(a, b, c) vst1_lane_s32(a, b, c)
#define VST1Q_LANE_I32(a, b, c) vst1q_lane_s32(a, b, c)
#define VST2_LANE_I32(a, b, c) vst2_lane_s32(a, b, c)
#define VST2Q_LANE_I32(a, b, c) vst2q_lane_s32(a, b, c)
#define VST3_LANE_I32(a, b, c) vst3_lane_s32(a, b, c)
#define VST3Q_LANE_I32(a, b, c) vst3q_lane_s32(a, b, c)
#define VST4_LANE_I32(a, b, c) vst4_lane_s32(a, b, c)
#define VST4Q_LANE_I32(a, b, c) vst4q_lane_s32(a, b, c)
#define VST1_I16(a, b) vst1_s16(a, b)
#define VST1Q_I16(a, b) vst1q_s16(a, b)
#define VST1_LANE_I16(a, b, c) vst1_lane_s16(a, b, c)
#define VST1Q_LANE_I16(a, b, c) vst1q_lane_s16(a, b, c)
#define VST2_LANE_I16(a, b, c) vst2_lane_s16(a, b, c)
#define VST2Q_LANE_I16(a, b, c) vst2q_lane_s16(a, b, c)
#define VST3_LANE_I16(a, b, c) vst3_lane_s16(a, b, c)
#define VST3Q_LANE_I16(a, b, c) vst3q_lane_s16(a, b, c)
#define VST4_LANE_I16(a, b, c) vst4_lane_s16(a, b, c)
#define VST4Q_LANE_I16(a, b, c) vst4q_lane_s16(a, b, c)
#define VMLAL_LANE_I16(a, b, c, d) vmlal_lane_s16(a, b, c, d)
#define VMLAL_HIGH_LANE_I16(a, b, c, d) vmlal_high_lane_s16(a, b, c, d)
#define VMLAL_N_I16(a, b, c) vmlal_n_s16(a, b, c)
#define VMLAL_HIGH_N_I16(a, b, c) vmlal_high_n_s16(a, b, c)
#define VMLAL_I16(a, b, c) vmlal_s16(a, b, c)
#define VPADAL_I16(a, b) vpadal_s16(a, b)
#define VPADALQ_I16(a, b) vpadalq_s16(a, b)
#define VPADD_I32(a, b) vpadd_s32(a, b)
#define VMULL_I8(a, b) vmull_s8(a, b)
#define VMULL_HIGH_I8(a, b) vmull_high_s8(a, b)
#define VGET_LOW_I8(a) vget_low_s8(a)
#define VGET_HIGH_I8(a) vget_high_s8(a)
#define VGET_LOW_I16(a) vget_low_s16(a)
#define VGET_HIGH_I16(a) vget_high_s16(a)
#define VGET_LANE_I16(a, b) vget_lane_s16(a, b)
#define VGETQ_LANE_I16(a, b) vgetq_lane_s16(a, b)
#define VGET_LOW_I32(a) vget_low_s32(a)
#define VGET_HIGH_I32(a) vget_high_s32(a)
#define VGET_LANE_I32(a, b) vget_lane_s32(a, b)
#define VGETQ_LANE_I32(a, b) vgetq_lane_s32(a, b)
#define VDUP_N_I32(a) vdup_n_s32(a)
#define VDUPQ_N_I32(a) vdupq_n_s32(a)
#define VDUP_N_I16(a) vdup_n_s16(a)
#define VDUPQ_N_I16(a) vdupq_n_s16(a)
#define VDUP_N_I8(a) vdup_n_s8(a)
#define VDUPQ_N_I8(a) vdupq_n_s8(a)
#define VSET_LANE_I32(a, b, c) vset_lane_s32(a, b, c)
#define VSETQ_LANE_I32(a, b, c) vsetq_lane_s32(a, b, c)
#define VSET_LANE_I16(a, b, c) vset_lane_s16(a, b, c)
#define VSETQ_LANE_I16(a, b, c) vsetq_lane_s16(a, b, c)
#define VZIP_I16(a, b) vzip_s16(a, b)
#define VUZP_I16(a, b) vuzp_s16(a, b)
#define VZIPQ_I32(a, b) vzipq_s32(a, b)
#define VZIP1_I32(a, b) vzip1_s32(a, b)
#define VZIP2_I32(a, b) vzip2_s32(a, b)
#define VZIP1Q_I32(a, b) vzip1q_s32(a, b)
#define VZIP2Q_I32(a, b) vzip2q_s32(a, b)
#define VZIP1Q_I64(a, b) vzip1q_s64(a, b)
#define VZIP2Q_I64(a, b) vzip2q_s64(a, b)
#define VTRN_I16(a, b) vtrn_s16(a, b)
#define VADD_I32(a, b) vadd_s32(a, b)
#define VADDQ_I32(a, b) vaddq_s32(a, b)
#define VCOMBINE_I32(a, b) vcombine_s32(a, b)
#define VDOT_I32(a, b, c) vdot_s32(a, b, c)
#define VDOT_LANE_I32(a, b, c, d) vdot_lane_s32(a, b, c, d)
#define VDOTQ_I32(a, b, c) vdotq_s32(a, b, c)
#define VDOTQ_LANE_I32(a, b, c, d) vdotq_lane_s32(a, b, c, d)
#define VDOTQ_LANEQ_I32(a, b, c, d) vdotq_laneq_s32(a, b, c, d)
#define VREINTERPRETQ_I32_I64(a) vreinterpretq_s32_s64(a)
#define VREINTERPRETQ_I64_I32(a) vreinterpretq_s64_s32(a)
#define VREINTERPRET_I8_I32(a) vreinterpret_s8_s32(a)
#define VREINTERPRETQ_I8_I32(a) vreinterpretq_s8_s32(a)
#define VREINTERPRET_I32_I8(a) vreinterpret_s32_s8(a)
#define VREINTERPRETQ_I32_I8(a) vreinterpretq_s32_s8(a)
#endif

#ifndef GEMM_UNSIGNED_INT
#define I8I32MLAGEMM s8s32mlagemm
#define I8I32MLAGEMM_SKINNYGER_ASCALAR s8s32mlagemm_skinnyger_ascalar
#define I8I32MLAGEMM_SKINNYGER_BSCALAR s8s32mlagemm_skinnyger_bscalar
#define I8I32MLAGEMM_SKINNYGER_CSCALAR s8s32mlagemm_skinnyger_cscalar
#define I8I32MLAGEMM_SKINNYGER_AVEC1 s8s32mlagemm_skinnyger_avec1
#define I8I32MLAGEMM_SKINNYGER_BVEC1 s8s32mlagemm_skinnyger_bvec1
#define I8I32MLAGEMM_SKINNYGER_CVEC1 s8s32mlagemm_skinnyger_cvec1
#define I8I32MLAGEMM_SKINNYGER_AVEC2 s8s32mlagemm_skinnyger_avec2
#define I8I32MLAGEMM_SKINNYGER_BVEC2 s8s32mlagemm_skinnyger_bvec2
#define I8I32MLAGEMM_SKINNYGER_CVEC2 s8s32mlagemm_skinnyger_cvec2
#define I8I32MLAGEMM_SKINNYGER_AVEC4 s8s32mlagemm_skinnyger_avec4
#define I8I32MLAGEMM_SKINNYGER_BVEC4 s8s32mlagemm_skinnyger_bvec4
#define I8I32MLAGEMM_SKINNYGER_CVEC4 s8s32mlagemm_skinnyger_cvec4
#define I8I32MLAGEMM_SKINNYGER_AVEC8 s8s32mlagemm_skinnyger_avec8
#define I8I32MLAGEMM_SKINNYGER_BVEC8 s8s32mlagemm_skinnyger_bvec8
#define I8I32MLAGEMM_SKINNYGER_CVEC8 s8s32mlagemm_skinnyger_cvec8
#define I8I32MLAGEMM_SKINNYGER_AVEC16 s8s32mlagemm_skinnyger_avec16
#define I8I32MLAGEMM_SKINNYGER_BVEC16 s8s32mlagemm_skinnyger_bvec16
#define I8I32MLAGEMM_SKINNYGER_CVEC16 s8s32mlagemm_skinnyger_cvec16
#define I8I32MLAGEMM_SKINNYDOT_ASCALAR s8s32mlagemm_skinnydot_ascalar
#define I8I32MLAGEMM_SKINNYDOT_BSCALAR s8s32mlagemm_skinnydot_bscalar
#define I8I32MLAGEMM_SKINNYDOT_CSCALAR s8s32mlagemm_skinnydot_cscalar
#define I8I32MLAGEMM_SKINNYDOT_AVEC1 s8s32mlagemm_skinnydot_avec1
#define I8I32MLAGEMM_SKINNYDOT_BVEC1 s8s32mlagemm_skinnydot_bvec1
#define I8I32MLAGEMM_SKINNYDOT_CVEC1 s8s32mlagemm_skinnydot_cvec1
#define I8I32MLAGEMM_SKINNYDOT_AVEC2 s8s32mlagemm_skinnydot_avec2
#define I8I32MLAGEMM_SKINNYDOT_BVEC2 s8s32mlagemm_skinnydot_bvec2
#define I8I32MLAGEMM_SKINNYDOT_CVEC2 s8s32mlagemm_skinnydot_cvec2
#define I8I32MLAGEMM_SKINNYDOT_AVEC4 s8s32mlagemm_skinnydot_avec4
#define I8I32MLAGEMM_SKINNYDOT_BVEC4 s8s32mlagemm_skinnydot_bvec4
#define I8I32MLAGEMM_SKINNYDOT_CVEC4 s8s32mlagemm_skinnydot_cvec4
#define I8I32MLAGEMM_SKINNYDOT_AVEC8 s8s32mlagemm_skinnydot_avec8
#define I8I32MLAGEMM_SKINNYDOT_BVEC8 s8s32mlagemm_skinnydot_bvec8
#define I8I32MLAGEMM_SKINNYDOT_CVEC8 s8s32mlagemm_skinnydot_cvec8
#define I8I32MLAGEMM_SKINNYDOT_AVEC16 s8s32mlagemm_skinnydot_avec16
#define I8I32MLAGEMM_SKINNYDOT_BVEC16 s8s32mlagemm_skinnydot_bvec16
#define I8I32MLAGEMM_SKINNYDOT_CVEC16 s8s32mlagemm_skinnydot_cvec16
#else
#define I8I32MLAGEMM u8u32mlagemm
#define I8I32MLAGEMM_SKINNYGER_ASCALAR u8u32mlagemm_skinnyger_ascalar
#define I8I32MLAGEMM_SKINNYGER_BSCALAR u8u32mlagemm_skinnyger_bscalar
#define I8I32MLAGEMM_SKINNYGER_CSCALAR u8u32mlagemm_skinnyger_cscalar
#define I8I32MLAGEMM_SKINNYGER_AVEC1 u8u32mlagemm_skinnyger_avec1
#define I8I32MLAGEMM_SKINNYGER_BVEC1 u8u32mlagemm_skinnyger_bvec1
#define I8I32MLAGEMM_SKINNYGER_CVEC1 u8u32mlagemm_skinnyger_cvec1
#define I8I32MLAGEMM_SKINNYGER_AVEC2 u8u32mlagemm_skinnyger_avec2
#define I8I32MLAGEMM_SKINNYGER_BVEC2 u8u32mlagemm_skinnyger_bvec2
#define I8I32MLAGEMM_SKINNYGER_CVEC2 u8u32mlagemm_skinnyger_cvec2
#define I8I32MLAGEMM_SKINNYGER_AVEC4 u8u32mlagemm_skinnyger_avec4
#define I8I32MLAGEMM_SKINNYGER_BVEC4 u8u32mlagemm_skinnyger_bvec4
#define I8I32MLAGEMM_SKINNYGER_CVEC4 u8u32mlagemm_skinnyger_cvec4
#define I8I32MLAGEMM_SKINNYGER_AVEC8 u8u32mlagemm_skinnyger_avec8
#define I8I32MLAGEMM_SKINNYGER_BVEC8 u8u32mlagemm_skinnyger_bvec8
#define I8I32MLAGEMM_SKINNYGER_CVEC8 u8u32mlagemm_skinnyger_cvec8
#define I8I32MLAGEMM_SKINNYGER_AVEC16 u8u32mlagemm_skinnyger_avec16
#define I8I32MLAGEMM_SKINNYGER_BVEC16 u8u32mlagemm_skinnyger_bvec16
#define I8I32MLAGEMM_SKINNYGER_CVEC16 u8u32mlagemm_skinnyger_cvec16
#define I8I32MLAGEMM_SKINNYDOT_ASCALAR u8u32mlagemm_skinnydot_ascalar
#define I8I32MLAGEMM_SKINNYDOT_BSCALAR u8u32mlagemm_skinnydot_bscalar
#define I8I32MLAGEMM_SKINNYDOT_CSCALAR u8u32mlagemm_skinnydot_cscalar
#define I8I32MLAGEMM_SKINNYDOT_AVEC1 u8u32mlagemm_skinnydot_avec1
#define I8I32MLAGEMM_SKINNYDOT_BVEC1 u8u32mlagemm_skinnydot_bvec1
#define I8I32MLAGEMM_SKINNYDOT_CVEC1 u8u32mlagemm_skinnydot_cvec1
#define I8I32MLAGEMM_SKINNYDOT_AVEC2 u8u32mlagemm_skinnydot_avec2
#define I8I32MLAGEMM_SKINNYDOT_BVEC2 u8u32mlagemm_skinnydot_bvec2
#define I8I32MLAGEMM_SKINNYDOT_CVEC2 u8u32mlagemm_skinnydot_cvec2
#define I8I32MLAGEMM_SKINNYDOT_AVEC4 u8u32mlagemm_skinnydot_avec4
#define I8I32MLAGEMM_SKINNYDOT_BVEC4 u8u32mlagemm_skinnydot_bvec4
#define I8I32MLAGEMM_SKINNYDOT_CVEC4 u8u32mlagemm_skinnydot_cvec4
#define I8I32MLAGEMM_SKINNYDOT_AVEC8 u8u32mlagemm_skinnydot_avec8
#define I8I32MLAGEMM_SKINNYDOT_BVEC8 u8u32mlagemm_skinnydot_bvec8
#define I8I32MLAGEMM_SKINNYDOT_CVEC8 u8u32mlagemm_skinnydot_cvec8
#define I8I32MLAGEMM_SKINNYDOT_AVEC16 u8u32mlagemm_skinnydot_avec16
#define I8I32MLAGEMM_SKINNYDOT_BVEC16 u8u32mlagemm_skinnydot_bvec16
#define I8I32MLAGEMM_SKINNYDOT_CVEC16 u8u32mlagemm_skinnydot_cvec16
#endif

#ifndef GEMM_UNSIGNED_INT
#define I8I32DOTGEMM s8s32dotgemm
#define I8I32DOTGEMM_SKINNYDOT_ASCALAR s8s32dotgemm_skinnydot_ascalar
#define I8I32DOTGEMM_SKINNYDOT_BSCALAR s8s32dotgemm_skinnydot_bscalar
#define I8I32DOTGEMM_SKINNYDOT_CSCALAR s8s32dotgemm_skinnydot_cscalar
#define I8I32DOTGEMM_SKINNYDOT_AVEC1 s8s32dotgemm_skinnydot_avec1
#define I8I32DOTGEMM_SKINNYDOT_BVEC1 s8s32dotgemm_skinnydot_bvec1
#define I8I32DOTGEMM_SKINNYDOT_CVEC1 s8s32dotgemm_skinnydot_cvec1
#define I8I32DOTGEMM_SKINNYDOT_AVEC2 s8s32dotgemm_skinnydot_avec2
#define I8I32DOTGEMM_SKINNYDOT_BVEC2 s8s32dotgemm_skinnydot_bvec2
#define I8I32DOTGEMM_SKINNYDOT_CVEC2 s8s32dotgemm_skinnydot_cvec2
#define I8I32DOTGEMM_SKINNYDOT_AVEC4 s8s32dotgemm_skinnydot_avec4
#define I8I32DOTGEMM_SKINNYDOT_BVEC4 s8s32dotgemm_skinnydot_bvec4
#define I8I32DOTGEMM_SKINNYDOT_CVEC4 s8s32dotgemm_skinnydot_cvec4
#define I8I32DOTGEMM_SKINNYDOT_AVEC8 s8s32dotgemm_skinnydot_avec8
#define I8I32DOTGEMM_SKINNYDOT_BVEC8 s8s32dotgemm_skinnydot_bvec8
#define I8I32DOTGEMM_SKINNYDOT_CVEC8 s8s32dotgemm_skinnydot_cvec8
#define I8I32DOTGEMM_SKINNYDOT_AVEC16 s8s32dotgemm_skinnydot_avec16
#define I8I32DOTGEMM_SKINNYDOT_BVEC16 s8s32dotgemm_skinnydot_bvec16
#define I8I32DOTGEMM_SKINNYDOT_CVEC16 s8s32dotgemm_skinnydot_cvec16
#else
#define I8I32DOTGEMM u8u32dotgemm
#define I8I32DOTGEMM_SKINNYDOT_ASCALAR u8u32dotgemm_skinnydot_ascalar
#define I8I32DOTGEMM_SKINNYDOT_BSCALAR u8u32dotgemm_skinnydot_bscalar
#define I8I32DOTGEMM_SKINNYDOT_CSCALAR u8u32dotgemm_skinnydot_cscalar
#define I8I32DOTGEMM_SKINNYDOT_AVEC1 u8u32dotgemm_skinnydot_avec1
#define I8I32DOTGEMM_SKINNYDOT_BVEC1 u8u32dotgemm_skinnydot_bvec1
#define I8I32DOTGEMM_SKINNYDOT_CVEC1 u8u32dotgemm_skinnydot_cvec1
#define I8I32DOTGEMM_SKINNYDOT_AVEC2 u8u32dotgemm_skinnydot_avec2
#define I8I32DOTGEMM_SKINNYDOT_BVEC2 u8u32dotgemm_skinnydot_bvec2
#define I8I32DOTGEMM_SKINNYDOT_CVEC2 u8u32dotgemm_skinnydot_cvec2
#define I8I32DOTGEMM_SKINNYDOT_AVEC4 u8u32dotgemm_skinnydot_avec4
#define I8I32DOTGEMM_SKINNYDOT_BVEC4 u8u32dotgemm_skinnydot_bvec4
#define I8I32DOTGEMM_SKINNYDOT_CVEC4 u8u32dotgemm_skinnydot_cvec4
#define I8I32DOTGEMM_SKINNYDOT_AVEC8 u8u32dotgemm_skinnydot_avec8
#define I8I32DOTGEMM_SKINNYDOT_BVEC8 u8u32dotgemm_skinnydot_bvec8
#define I8I32DOTGEMM_SKINNYDOT_CVEC8 u8u32dotgemm_skinnydot_cvec8
#define I8I32DOTGEMM_SKINNYDOT_AVEC16 u8u32dotgemm_skinnydot_avec16
#define I8I32DOTGEMM_SKINNYDOT_BVEC16 u8u32dotgemm_skinnydot_bvec16
#define I8I32DOTGEMM_SKINNYDOT_CVEC16 u8u32dotgemm_skinnydot_cvec16
#endif

#endif
