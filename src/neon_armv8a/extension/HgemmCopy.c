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


#include "common/CommonCopy.h"
#include <arm_neon.h>

static inline void pref_ab(const float16_t *dat) {
  __asm__ ("prfm pldl1keep,[%0,#64]\n\t"::"r"(dat):);
}

#define NCOPY_NEON_LOOP_K8_UNROLL4(inc, dst_ptr, src1, src2, src3, src4) \
  for (dim1_count = dim1_cache; dim1_count > 7; dim1_count -= 8) {\
    float16x8x4_t t1;\
    t1.val[0] = vld1q_f16(src1); src1 += 8; pref_ab(src1);\
    t1.val[1] = vld1q_f16(src2); src2 += 8; pref_ab(src2);\
    t1.val[2] = vld1q_f16(src3); src3 += 8; pref_ab(src3);\
    t1.val[3] = vld1q_f16(src4); src4 += 8; pref_ab(src4);\
    vst4q_lane_f16(dst_ptr, t1, 0);\
    vst4q_lane_f16(dst_ptr + inc, t1, 1);\
    vst4q_lane_f16(dst_ptr + inc * 2, t1, 2);\
    vst4q_lane_f16(dst_ptr + inc * 3, t1, 3);\
    vst4q_lane_f16(dst_ptr + inc * 4, t1, 4);\
    vst4q_lane_f16(dst_ptr + inc * 5, t1, 5);\
    vst4q_lane_f16(dst_ptr + inc * 6, t1, 6);\
    vst4q_lane_f16(dst_ptr + inc * 7, t1, 7);\
    dst_ptr += inc * 8;\
  }

#define NCOPY_UNROLL_16 {\
  float16_t *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  NCOPY_NEON_LOOP_K8_UNROLL4(16, dst_h1, src1, src2, src3, src4)\
  dst_h1 = dst1 + 4;\
  NCOPY_NEON_LOOP_K8_UNROLL4(16, dst_h1, src5, src6, src7, src8)\
  dst_h1 = dst1 + 8;\
  NCOPY_NEON_LOOP_K8_UNROLL4(16, dst_h1, src9, src10, src11, src12)\
  dst_h1 = dst1 + 12;\
  NCOPY_NEON_LOOP_K8_UNROLL4(16, dst_h1, src13, src14, src15, src16)\
  dst1 = dst_h1 - 12;\
  NCOPY_STD(16)\
}

#define NCOPY_UNROLL_8 {\
  float16_t *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  NCOPY_NEON_LOOP_K8_UNROLL4(8, dst_h1, src1, src2, src3, src4)\
  dst_h1 = dst1 + 4;\
  NCOPY_NEON_LOOP_K8_UNROLL4(8, dst_h1, src5, src6, src7, src8)\
  dst1 = dst_h1 - 4;\
  NCOPY_STD(8)\
}

#define NCOPY_UNROLL_4 {\
  float16_t *dst_h1 = dst1; uint32_t dim1_cache = dim1_count;\
  NCOPY_NEON_LOOP_K8_UNROLL4(4, dst_h1, src1, src2, src3, src4)\
  dst1 = dst_h1;\
  NCOPY_STD(4)\
}

#define NCOPY_UNROLL_2 NCOPY_STD(2)
#define NCOPY_UNROLL_1 NCOPY_STD(1)

#define NCOPY_float16_t_float16_t(unroll) NCOPY_UNROLL_##unroll


#define TCOPY_UNIT_1(src_ptr, dst_ptr, dst_offset) \
  dst_ptr[dst_offset] = *src_ptr;

#define TCOPY_UNIT_2(src_ptr, dst_ptr, dst_offset) {\
  dst_ptr[dst_offset] = *src_ptr;\
  dst_ptr[dst_offset + 1] = src_ptr[1];\
}

#define TCOPY_UNIT_4(src_ptr, dst_ptr, dst_offset) {\
  float16x4_t tmp = vld1_f16(src_ptr); pref_ab(src_ptr + 4);\
  vst1_f16(dst_ptr + dst_offset, tmp);\
}

#define TCOPY_UNIT_8(src_ptr, dst_ptr, dst_offset) {\
  float16x8_t tmp = vld1q_f16(src_ptr); pref_ab(src_ptr + 8);\
  vst1q_f16(dst_ptr + dst_offset, tmp);\
}

#define TCOPY_UNIT_16(src_ptr, dst_ptr, dst_offset) {\
  float16x8_t tmp1 = vld1q_f16(src_ptr);\
  float16x8_t tmp2 = vld1q_f16(src_ptr + 8); pref_ab(src_ptr + 16);\
  vst1q_f16(dst_ptr + dst_offset, tmp1);\
  vst1q_f16(dst_ptr + dst_offset + 8, tmp2);\
}

#define TCOPY_UNIT_float16_t_float16_t(src_ptr, dst_ptr, dst_offset, num_elements) \
  TCOPY_UNIT_##num_elements(src_ptr, dst_ptr, dst_offset)

GENERIC_NCOPY_FUNC(hgemm, float16_t, float16_t, 8)
GENERIC_NCOPY_FUNC(hgemm, float16_t, float16_t, 16)

GENERIC_TCOPY_FUNC(hgemm, float16_t, float16_t, 8)
GENERIC_TCOPY_FUNC(hgemm, float16_t, float16_t, 16)

