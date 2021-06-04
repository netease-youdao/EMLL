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
 * File:        NeonI8I32MlaGemmCopy.h
 * Description: Source code template for NEON 8->32bit GEMM packing functions
 ****************************************************************************/

#include "NeonIntOpSign.h"

#ifndef INCLUDE_NEON_I8I32_COPY
#define INCLUDE_NEON_I8I32_COPY

static inline void pref_ab(const I8 *dat) {
#if __aarch64__
  __asm__ ("prfm pldl1keep,[%0,#64]\n\t"::"r"(dat):);
#else
  __asm__ ("pld [%0,#64]\n\t"::"r"(dat):);
#endif
}

#define NCOPY_LOOP_K8_UNROLL4(inc, dst_ptr, src1, src2, src3, src4) \
  for (dim1_count = dim1_cache; dim1_count > 7; dim1_count -= 8) {\
    I8X8 d1 = VLD1_I8(src1); src1 += 8; pref_ab(src1);\
    I8X8 d2 = VLD1_I8(src2); src2 += 8; pref_ab(src2);\
    I8X8 d3 = VLD1_I8(src3); src3 += 8; pref_ab(src3);\
    I8X8 d4 = VLD1_I8(src4); src4 += 8; pref_ab(src4);\
    I16X8X4 tm1;\
    tm1.val[0] = VMOVL_I8(d1); tm1.val[1] = VMOVL_I8(d2);\
    tm1.val[2] = VMOVL_I8(d3); tm1.val[3] = VMOVL_I8(d4);\
    VST4Q_LANE_I16(dst_ptr, tm1, 0);\
    VST4Q_LANE_I16(dst_ptr + inc, tm1, 1);\
    VST4Q_LANE_I16(dst_ptr + inc * 2, tm1, 2);\
    VST4Q_LANE_I16(dst_ptr + inc * 3, tm1, 3);\
    VST4Q_LANE_I16(dst_ptr + inc * 4, tm1, 4);\
    VST4Q_LANE_I16(dst_ptr + inc * 5, tm1, 5);\
    VST4Q_LANE_I16(dst_ptr + inc * 6, tm1, 6);\
    VST4Q_LANE_I16(dst_ptr + inc * 7, tm1, 7);\
    dst_ptr += inc * 8;\
  }

#define NCOPY_LOOP_K8_UNROLL3(inc, dst_ptr, src1, src2, src3) \
  for (dim1_count = dim1_cache; dim1_count > 7; dim1_count -= 8) {\
    I8X8 d1 = VLD1_I8(src1); src1 += 8; pref_ab(src1);\
    I8X8 d2 = VLD1_I8(src2); src2 += 8; pref_ab(src2);\
    I8X8 d3 = VLD1_I8(src3); src3 += 8; pref_ab(src3);\
    I16X8X3 tm1;\
    tm1.val[0] = VMOVL_I8(d1);\
    tm1.val[1] = VMOVL_I8(d2);\
    tm1.val[2] = VMOVL_I8(d3);\
    VST3Q_LANE_I16(dst_ptr, tm1, 0);\
    VST3Q_LANE_I16(dst_ptr + inc, tm1, 1);\
    VST3Q_LANE_I16(dst_ptr + inc * 2, tm1, 2);\
    VST3Q_LANE_I16(dst_ptr + inc * 3, tm1, 3);\
    VST3Q_LANE_I16(dst_ptr + inc * 4, tm1, 4);\
    VST3Q_LANE_I16(dst_ptr + inc * 5, tm1, 5);\
    VST3Q_LANE_I16(dst_ptr + inc * 6, tm1, 6);\
    VST3Q_LANE_I16(dst_ptr + inc * 7, tm1, 7);\
    dst_ptr += inc * 8;\
  }

#define NCOPY_UNROLL_12 {\
  I16 *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  NCOPY_LOOP_K8_UNROLL4(12, dst_h1, src1, src2, src3, src4)\
  dst_h1 = dst1 + 4;\
  NCOPY_LOOP_K8_UNROLL4(12, dst_h1, src5, src6, src7, src8)\
  dst_h1 = dst1 + 8;\
  NCOPY_LOOP_K8_UNROLL4(12, dst_h1, src9, src10, src11, src12)\
  dst1 = dst_h1 - 8;\
  NCOPY_STD(12)\
}

#define NCOPY_UNROLL_8 {\
  I16 *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  NCOPY_LOOP_K8_UNROLL4(8, dst_h1, src1, src2, src3, src4)\
  dst_h1 = dst1 + 4;\
  NCOPY_LOOP_K8_UNROLL4(8, dst_h1, src5, src6, src7, src8)\
  dst1 = dst_h1 - 4;\
  NCOPY_STD(8)\
}

#define NCOPY_UNROLL_6 {\
  I16 *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  NCOPY_LOOP_K8_UNROLL3(6, dst_h1, src1, src2, src3)\
  dst_h1 = dst1 + 3;\
  NCOPY_LOOP_K8_UNROLL3(6, dst_h1, src4, src5, src6)\
  dst1 = dst_h1 - 3;\
  NCOPY_STD(6)\
}

#define NCOPY_UNROLL_4 {\
  uint32_t dim1_cache = dim1_count;\
  NCOPY_LOOP_K8_UNROLL4(4, dst1, src1, src2, src3, src4)\
  NCOPY_STD(4)\
}

#define NCOPY_UNROLL_2 NCOPY_STD(2)
#define NCOPY_UNROLL_1 NCOPY_STD(1)

#ifdef GEMM_UNSIGNED_INT
#define NCOPY_uint8_t_uint16_t(unroll) NCOPY_UNROLL_##unroll
#else
#define NCOPY_int8_t_int16_t(unroll) NCOPY_UNROLL_##unroll
#endif

#define TCOPY_UNIT_1(src_ptr, dst_ptr, dst_offset) \
  TCOPY_UNIT_STD(src_ptr, dst_ptr, dst_offset, 1)

#define TCOPY_UNIT_2(src_ptr, dst_ptr, dst_offset) \
  TCOPY_UNIT_STD(src_ptr, dst_ptr, dst_offset, 2)

static inline I16X4 vld1_i16_i8(const I8 *src) {
#if __aarch64__
  I16X4 ret;
  __asm__("ldr %s0,[%1]; "ISHLL" %0.8h,%0.8b,#0\n\t"
    :"=w"(ret):"r"(src):"memory","cc");
  return ret;
#else
  I16X8 ret;
  __asm__("vld1.32 {d0[0]},[%1]; "ASM_VMOVL_I8" %q0,d0\n\t"
    :"=w"(ret):"r"(src):"memory","cc","d0");
  return VGET_LOW_I16(ret);
#endif
}

static inline I16X8 vld1q_i16_i8(const I8 *src) {
  return VMOVL_I8(VLD1_I8(src));
}

#define TCOPY_UNIT_4(src_ptr, dst_ptr, dst_offset) {\
  I16X4 tmp = vld1_i16_i8(src_ptr);\
  VST1_I16(dst_ptr + dst_offset, tmp);\
}

#define TCOPY_UNIT_6(src_ptr, dst_ptr, dst_offset) {\
  I16X4 tmp = vld1_i16_i8(src_ptr);\
  I16 t5 = src_ptr[4];\
  I16 t6 = src_ptr[5];\
  pref_ab(src_ptr + 6);\
  VST1_I16(dst_ptr + dst_offset, tmp);\
  dst_ptr[dst_offset + 4] = t5;\
  dst_ptr[dst_offset + 5] = t6;\
}

#define TCOPY_UNIT_8(src_ptr, dst_ptr, dst_offset) {\
  I16X8 tmp = vld1q_i16_i8(src_ptr);\
  pref_ab(src_ptr + 8);\
  VST1Q_I16(dst_ptr + dst_offset, tmp);\
}

#define TCOPY_UNIT_12(src_ptr, dst_ptr, dst_offset) {\
  I16X8 tmpq = vld1q_i16_i8(src_ptr);\
  I16X4 tmpd = vld1_i16_i8(src_ptr + 8);\
  pref_ab(src_ptr + 12);\
  VST1Q_I16(dst_ptr + dst_offset, tmpq);\
  VST1_I16(dst_ptr + dst_offset + 8, tmpd);\
}

#ifdef GEMM_UNSIGNED_INT
#define TCOPY_UNIT_uint8_t_uint16_t(src_ptr, dst_ptr, dst_offset, num_elements) \
  TCOPY_UNIT_##num_elements(src_ptr, dst_ptr, dst_offset)
#else
#define TCOPY_UNIT_int8_t_int16_t(src_ptr, dst_ptr, dst_offset, num_elements) \
  TCOPY_UNIT_##num_elements(src_ptr, dst_ptr, dst_offset)
#endif

#endif
