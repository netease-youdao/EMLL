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
 * File:        NeonSgemmCopy.h
 * Description: Code templates for NEON SGEMM packing functions.
 *****************************************************************************/

#include <arm_neon.h>

#ifndef INCLUDE_NEON_SGEMM_COPY
#define INCLUDE_NEON_SGEMM_COPY

#if __aarch64__
static inline void pref_ab(const float *dat) {
  __asm__ ("prfm pldl1keep,[%0,#64]\n\t"::"r"(dat):);
}
#else
static inline void pref_ab(const float *dat) {
  __asm__ ("pld [%0,#64]\n\t"::"r"(dat):);
}
#endif

#define NCOPY_NEON_LOOP_K8_UNROLL4(inc, dst_ptr, src1, src2, src3, src4) \
  for (dim1_count = dim1_cache; dim1_count > 7; dim1_count -= 8) {\
    t1.val[0] = vld1q_f32(src1); t2.val[0] = vld1q_f32(src1 + 4);\
    src1 += 8; pref_ab(src1);\
    t1.val[1] = vld1q_f32(src2); t2.val[1] = vld1q_f32(src2 + 4);\
    src2 += 8; pref_ab(src2);\
    t1.val[2] = vld1q_f32(src3); t2.val[2] = vld1q_f32(src3 + 4);\
    src3 += 8; pref_ab(src3);\
    t1.val[3] = vld1q_f32(src4); t2.val[3] = vld1q_f32(src4 + 4);\
    src4 += 8; pref_ab(src4);\
    vst4q_lane_f32(dst_ptr, t1, 0);\
    vst4q_lane_f32(dst_ptr + inc, t1, 1);\
    vst4q_lane_f32(dst_ptr + inc * 2, t1, 2);\
    vst4q_lane_f32(dst_ptr + inc * 3, t1, 3);\
    vst4q_lane_f32(dst_ptr + inc * 4, t2, 0);\
    vst4q_lane_f32(dst_ptr + inc * 5, t2, 1);\
    vst4q_lane_f32(dst_ptr + inc * 6, t2, 2);\
    vst4q_lane_f32(dst_ptr + inc * 7, t2, 3);\
    dst_ptr += inc * 8;\
  }\


#define NCOPY_UNROLL_24 {\
  float32x4x4_t t1, t2;\
  float *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  NCOPY_NEON_LOOP_K8_UNROLL4(24, dst_h1, src1, src2, src3, src4)\
  dst_h1 = dst1 + 4;\
  NCOPY_NEON_LOOP_K8_UNROLL4(24, dst_h1, src5, src6, src7, src8)\
  dst_h1 = dst1 + 8;\
  NCOPY_NEON_LOOP_K8_UNROLL4(24, dst_h1, src9, src10, src11, src12)\
  dst_h1 = dst1 + 12;\
  NCOPY_NEON_LOOP_K8_UNROLL4(24, dst_h1, src13, src14, src15, src16)\
  dst_h1 = dst1 + 16;\
  NCOPY_NEON_LOOP_K8_UNROLL4(24, dst_h1, src17, src18, src19, src20)\
  dst_h1 = dst1 + 20;\
  NCOPY_NEON_LOOP_K8_UNROLL4(24, dst_h1, src21, src22, src23, src24)\
  dst1 = dst_h1 - 20;\
  NCOPY_STD(24)\
}

#define NCOPY_UNROLL_12 {\
  float32x4x4_t t1, t2;\
  float *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  NCOPY_NEON_LOOP_K8_UNROLL4(12, dst_h1, src1, src2, src3, src4)\
  dst_h1 = dst1 + 4;\
  NCOPY_NEON_LOOP_K8_UNROLL4(12, dst_h1, src5, src6, src7, src8)\
  dst_h1 = dst1 + 8;\
  NCOPY_NEON_LOOP_K8_UNROLL4(12, dst_h1, src9, src10, src11, src12)\
  dst1 = dst_h1 - 8;\
  NCOPY_STD(12)\
}

#define NCOPY_UNROLL_8 {\
  float32x4x4_t t1, t2;\
  float *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  NCOPY_NEON_LOOP_K8_UNROLL4(8, dst_h1, src1, src2, src3, src4)\
  dst_h1 = dst1 + 4;\
  NCOPY_NEON_LOOP_K8_UNROLL4(8, dst_h1, src5, src6, src7, src8)\
  dst1 = dst_h1 - 4;\
  NCOPY_STD(8)\
}

#define NCOPY_UNROLL_6 {\
  float32x4x3_t t1, t2;\
  float *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  for (; dim1_count > 7; dim1_count -= 8) {\
    t1.val[0] = vld1q_f32(src1); t2.val[0] = vld1q_f32(src1 + 4);\
    src1 += 8; pref_ab(src1);\
    t1.val[1] = vld1q_f32(src2); t2.val[1] = vld1q_f32(src2 + 4);\
    src2 += 8; pref_ab(src2);\
    t1.val[2] = vld1q_f32(src3); t2.val[2] = vld1q_f32(src3 + 4);\
    src3 += 8; pref_ab(src3);\
    vst3q_lane_f32(dst_h1, t1, 0);\
    vst3q_lane_f32(dst_h1 + 6, t1, 1);\
    vst3q_lane_f32(dst_h1 + 12, t1, 2);\
    vst3q_lane_f32(dst_h1 + 18, t1, 3);\
    vst3q_lane_f32(dst_h1 + 24, t2, 0);\
    vst3q_lane_f32(dst_h1 + 30, t2, 1);\
    vst3q_lane_f32(dst_h1 + 36, t2, 2);\
    vst3q_lane_f32(dst_h1 + 42, t2, 3);\
    dst_h1 += 48;\
  }\
  float *dst_h2 = dst1 + 3;\
  for (dim1_count = dim1_cache; dim1_count > 7; dim1_count -= 8) {\
    t1.val[0] = vld1q_f32(src4); t2.val[0] = vld1q_f32(src4 + 4);\
    src4 += 8; pref_ab(src4);\
    t1.val[1] = vld1q_f32(src5); t2.val[1] = vld1q_f32(src5 + 4);\
    src5 += 8; pref_ab(src5);\
    t1.val[2] = vld1q_f32(src6); t2.val[2] = vld1q_f32(src6 + 4);\
    src6 += 8; pref_ab(src6);\
    vst3q_lane_f32(dst_h2, t1, 0);\
    vst3q_lane_f32(dst_h2 + 6, t1, 1);\
    vst3q_lane_f32(dst_h2 + 12, t1, 2);\
    vst3q_lane_f32(dst_h2 + 18, t1, 3);\
    vst3q_lane_f32(dst_h2 + 24, t2, 0);\
    vst3q_lane_f32(dst_h2 + 30, t2, 1);\
    vst3q_lane_f32(dst_h2 + 36, t2, 2);\
    vst3q_lane_f32(dst_h2 + 42, t2, 3);\
    dst_h2 += 48;\
  }\
  dst1 = dst_h1;\
  NCOPY_STD(6)\
}

#define NCOPY_UNROLL_4 {\
  float32x4x4_t t1;\
  for (; dim1_count > 3; dim1_count -= 4) {\
    t1.val[0] = vld1q_f32(src1); src1 += 4; pref_ab(src1);\
    t1.val[1] = vld1q_f32(src2); src2 += 4; pref_ab(src2);\
    t1.val[2] = vld1q_f32(src3); src3 += 4; pref_ab(src3);\
    t1.val[3] = vld1q_f32(src4); src4 += 4; pref_ab(src4);\
    vst4q_f32(dst1,t1); dst1 += 16;\
  }\
  NCOPY_STD(4)\
}

#define NCOPY_UNROLL_2 NCOPY_STD(2)
#define NCOPY_UNROLL_1 NCOPY_STD(1)

//#define NCOPY_a(unroll) NCOPY_UNROLL_##unroll
//#define NCOPY_b(unroll) NCOPY_UNROLL_##unroll

#define TCOPY_UNIT_1(src_ptr, dst_ptr, dst_offset) \
  dst_ptr[dst_offset] = *src_ptr;

#define TCOPY_UNIT_2(src_ptr, dst_ptr, dst_offset) {\
  float32x2_t tmp = vld1_f32(src_ptr);\
  vst1_f32(dst_ptr + dst_offset, tmp);\
}

#define TCOPY_UNIT_4(src_ptr, dst_ptr, dst_offset) {\
  float32x4_t tmp = vld1q_f32(src_ptr); pref_ab(src_ptr + 4);\
  vst1q_f32(dst_ptr + dst_offset, tmp);\
}

#define TCOPY_UNIT_6(src_ptr, dst_ptr, dst_offset) {\
  float32x4_t tmpq = vld1q_f32(src_ptr);\
  float32x2_t tmpd = vld1_f32(src_ptr + 4); pref_ab(src_ptr + 6);\
  vst1q_f32(dst_ptr + dst_offset, tmpq);\
  vst1_f32(dst_ptr + dst_offset + 4, tmpd);\
}

#define TCOPY_UNIT_8(src_ptr, dst_ptr, dst_offset) {\
  float32x4_t tmp1 = vld1q_f32(src_ptr);\
  float32x4_t tmp2 = vld1q_f32(src_ptr + 4); pref_ab(src_ptr + 8);\
  vst1q_f32(dst_ptr + dst_offset, tmp1);\
  vst1q_f32(dst_ptr + dst_offset + 4, tmp2);\
}

#define TCOPY_UNIT_12(src_ptr, dst_ptr, dst_offset) {\
  float32x4_t tmp1 = vld1q_f32(src_ptr);\
  float32x4_t tmp2 = vld1q_f32(src_ptr + 4);\
  float32x4_t tmp3 = vld1q_f32(src_ptr + 8); pref_ab(src_ptr + 12);\
  vst1q_f32(dst_ptr + dst_offset, tmp1);\
  vst1q_f32(dst_ptr + dst_offset + 4, tmp2);\
  vst1q_f32(dst_ptr + dst_offset + 8, tmp3);\
}

#define TCOPY_UNIT_24(src_ptr, dst_ptr, dst_offset) {\
  float32x4_t tmp1 = vld1q_f32(src_ptr);\
  float32x4_t tmp2 = vld1q_f32(src_ptr + 4);\
  float32x4_t tmp3 = vld1q_f32(src_ptr + 8);\
  float32x4_t tmp4 = vld1q_f32(src_ptr + 12);\
  float32x4_t tmp5 = vld1q_f32(src_ptr + 16); pref_ab(src_ptr + 24);\
  float32x4_t tmp6 = vld1q_f32(src_ptr + 20); pref_ab(src_ptr + 40);\
  vst1q_f32(dst_ptr + dst_offset, tmp1);\
  vst1q_f32(dst_ptr + dst_offset + 4, tmp2);\
  vst1q_f32(dst_ptr + dst_offset + 8, tmp3);\
  vst1q_f32(dst_ptr + dst_offset + 12, tmp4);\
  vst1q_f32(dst_ptr + dst_offset + 16, tmp5);\
  vst1q_f32(dst_ptr + dst_offset + 20, tmp6);\
}

//#define TCOPY_UNIT_a(src_ptr, dst_ptr, dst_offset, num_elements) \
  TCOPY_UNIT_##num_elements(src_ptr, dst_ptr, dst_offset)

//#define TCOPY_UNIT_b(src_ptr, dst_ptr, dst_offset, num_elements) \
  TCOPY_UNIT_##num_elements(src_ptr, dst_ptr, dst_offset)

#endif
