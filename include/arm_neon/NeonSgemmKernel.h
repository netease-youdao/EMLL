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
 * File:        NeonSgemmKernel.h
 * Description: Common building blocks for NEON SGEMM kernel functions.
 *****************************************************************************/

#include <stdint.h>
#include <arm_neon.h>

#ifndef INCLUDE_NEON_SGEMM_KERNEL
#define INCLUDE_NEON_SGEMM_KERNEL

#if __aarch64__

static inline void pref_c(float *dat) {
  __asm__ ("prfm pstl1keep,[%0]\n\t"::"r"(dat):);
}

#else

static inline void pref_c(float *dat) {
  __asm__ ("pld [%0]\n\t"::"r"(dat):);
}

#define vfmaq_lane_f32(c1,a1,b1,id) vmlaq_lane_f32(c1,a1,b1,id)
#define vfma_lane_f32(c1,a1,b1,id) vmla_lane_f32(c1,a1,b1,id)
#define vmlaq_laneq0_f32(c1,a1,b1) vmlaq_lane_f32(c1,a1,vget_low_f32(b1),0)
#define vmlaq_laneq1_f32(c1,a1,b1) vmlaq_lane_f32(c1,a1,vget_low_f32(b1),1)
#define vmlaq_laneq2_f32(c1,a1,b1) vmlaq_lane_f32(c1,a1,vget_high_f32(b1),0)
#define vmlaq_laneq3_f32(c1,a1,b1) vmlaq_lane_f32(c1,a1,vget_high_f32(b1),1)
#define vfmaq_laneq_f32(c1,a1,b1,laneid) vmlaq_laneq##laneid##_f32(c1,a1,b1)
#define vmla_laneq0_f32(c1,a1,b1) vmla_lane_f32(c1,a1,vget_low_f32(b1),0)
#define vmla_laneq1_f32(c1,a1,b1) vmla_lane_f32(c1,a1,vget_low_f32(b1),1)
#define vmla_laneq2_f32(c1,a1,b1) vmla_lane_f32(c1,a1,vget_high_f32(b1),0)
#define vmla_laneq3_f32(c1,a1,b1) vmla_lane_f32(c1,a1,vget_high_f32(b1),1)
#define vfma_laneq_f32(c1,a1,b1,laneid) vmla_laneq##laneid##_f32(c1,a1,b1)
#define vfma_n_f32(c1,a1,b1) vmla_n_f32(c1,a1,b1)
#define vfmaq_n_f32(c1,a1,b1) vmlaq_n_f32(c1,a1,b1)
#define vfma_f32(c1,a1,b1) vmla_f32(c1,a1,b1)
#define vfmaq_f32(c1,a1,b1) vmlaq_f32(c1,a1,b1)

#endif

#define NEON_SGEMM_KERNEL_M1N1 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  float32x2_t ad01, bd01;\
  float32x2_t cd01 = vdup_n_f32(0.0f);\
  uint32_t k_left = K;\
  if (k_left > 1) {\
    ad01 = vld1_f32(a_ptr); a_ptr += 2;\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  for (; k_left > 3; k_left-=2) {\
    cd01 = vfma_f32(cd01, ad01, bd01);\
    ad01 = vld1_f32(a_ptr); a_ptr += 2;\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  if (k_left > 1) {\
    cd01 = vfma_f32(cd01, ad01, bd01); k_left -= 2;\
  }\
  float cs01 = vget_lane_f32(cd01, 0) + vget_lane_f32(cd01, 1);\
  if (k_left > 0) {\
    cs01 += (*a_ptr) * (*b_ptr1); a_ptr++;\
  }
    
#define NEON_SGEMM_SAVE_M1N1 \
  cs01 += beta * (*c_ptr);\
  *c_ptr = cs01;

#define NEON_SGEMM_KERNEL_M2N1_UNIT(a_ptr1, b_ptr1) \
  float32x2_t ad01, ad02, bd01, cd01, cd02;\
  cd01 = cd02 = vdup_n_f32(0.0f);\
  uint32_t k_left = K;\
  if (k_left > 1) {\
    ad01 = vld1_f32(a_ptr1); ad02 = vld1_f32(a_ptr1 + 2); a_ptr1 += 4;\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cd01 = vfma_lane_f32(cd01, ad01, bd01, 0); ad01 = vld1_f32(a_ptr1);\
    cd02 = vfma_lane_f32(cd02, ad02, bd01, 1); ad02 = vld1_f32(a_ptr1 + 2);\
    a_ptr1 += 4; bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  if(k_left > 1) {\
    cd01 = vfma_lane_f32(cd01, ad01, bd01, 0);\
    cd02 = vfma_lane_f32(cd02, ad02, bd01, 1); k_left -= 2;\
  }\
  cd01 = vadd_f32(cd01, cd02);\
  if(k_left > 0) {\
    ad01 = vld1_f32(a_ptr1); a_ptr1 += 2;\
    cd01 = vfma_n_f32(cd01, ad01, *b_ptr1); b_ptr1++;\
  }

#define NEON_SGEMM_KERNEL_M2N1 \
  const float *b_ptr1 = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M2N1_UNIT(a_ptr, b_ptr1)

#define NEON_SGEMM_KERNEL_M1N2 \
  const float *b_ptr1 = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M2N1_UNIT(b_ptr1, a_ptr)

#define NEON_SGEMM_SAVE_M2N1 \
  float32x2_t ct1 = vld1_f32(c_ptr);\
  cd01 = vfma_n_f32(cd01, ct1, beta);\
  vst1_f32(c_ptr, cd01);

#define NEON_SGEMM_SAVE_M1N2_UNIT(cd01) \
  c_tmp[0] = c_tmp[0] * beta + vget_lane_f32(cd01, 0);\
  c_tmp[ldc] = c_tmp[ldc] * beta + vget_lane_f32(cd01, 1);\
  c_tmp += ldc * 2;

#define NEON_SGEMM_SAVE_M1N2 float *c_tmp = c_ptr; NEON_SGEMM_SAVE_M1N2_UNIT(cd01)

#define NEON_SGEMM_KERNEL_M2N2 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  float32x2_t ad01, ad02, bd01, bd02;\
  float32x2_t cd01, cd02, cd03, cd04;\
  cd01 = cd02 = cd03 = cd04 = vdup_n_f32(0.0f);\
  uint32_t k_left = K;\
  if (k_left > 1) {\
    ad01 = vld1_f32(a_ptr); ad02 = vld1_f32(a_ptr + 2); a_ptr += 4;\
    bd01 = vld1_f32(b_ptr1); bd02 = vld1_f32(b_ptr1 + 2); b_ptr1 += 4;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cd01 = vfma_lane_f32(cd01, ad01, bd01, 0);\
    cd02 = vfma_lane_f32(cd02, ad01, bd01, 1);\
    ad01 = vld1_f32(a_ptr); bd01 = vld1_f32(b_ptr1);\
    cd03 = vfma_lane_f32(cd03, ad02, bd02, 0);\
    cd04 = vfma_lane_f32(cd04, ad02, bd02, 1);\
    ad02 = vld1_f32(a_ptr + 2); a_ptr += 4;\
    bd02 = vld1_f32(b_ptr1 + 2); b_ptr1 += 4;\
  }\
  if (k_left > 1) {\
    cd01 = vfma_lane_f32(cd01, ad01, bd01, 0);\
    cd02 = vfma_lane_f32(cd02, ad01, bd01, 1);\
    cd03 = vfma_lane_f32(cd03, ad02, bd02, 0);\
    cd04 = vfma_lane_f32(cd04, ad02, bd02, 1); k_left -= 2;\
  }\
  cd01 = vadd_f32(cd01, cd03);\
  cd02 = vadd_f32(cd02, cd04);\
  if (k_left > 0) {\
    ad01 = vld1_f32(a_ptr); a_ptr += 2;\
    bd01 = vld1_f32(b_ptr1);\
    cd01 = vfma_lane_f32(cd01, ad01, bd01, 0);\
    cd02 = vfma_lane_f32(cd02, ad01, bd01, 1);\
  }

#define NEON_SGEMM_SAVE_M2N2_UNIT(cd01, cd02) \
  ct1 = vld1_f32(c_tmp);\
  ct2 = vld1_f32(c_tmp + ldc);\
  cd01 = vfma_n_f32(cd01, ct1, beta);\
  cd02 = vfma_n_f32(cd02, ct2, beta);\
  vst1_f32(c_tmp, cd01);\
  vst1_f32(c_tmp + ldc, cd02); c_tmp += ldc * 2;

#define NEON_SGEMM_SAVE_M2N2 \
  float *c_tmp = c_ptr;\
  float32x2_t ct1, ct2; NEON_SGEMM_SAVE_M2N2_UNIT(cd01, cd02)

#define NEON_SGEMM_KERNEL_M4N1_UNIT(a_ptr1, b_ptr1) \
  uint32_t k_left = K;\
  float32x4_t aq01, aq02, cq01, cq02;\
  float32x2_t bd01;\
  cq01 = cq02 = vdupq_n_f32(0.0f);\
  if (k_left > 1) {\
    aq01 = vld1q_f32(a_ptr1); aq02 = vld1q_f32(a_ptr1 + 4); a_ptr1 += 8;\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0); aq01 = vld1q_f32(a_ptr1);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 1); aq02 = vld1q_f32(a_ptr1 + 4);\
    a_ptr1 += 8; bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  if (k_left > 1) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 1);\
    k_left -= 2;\
  }\
  cq01 = vaddq_f32(cq01, cq02);\
  if (k_left > 0) {\
    aq01 = vld1q_f32(a_ptr1); a_ptr1 += 4;\
    cq01 = vfmaq_n_f32(cq01, aq01, *b_ptr1); b_ptr1++;\
  }

#define NEON_SGEMM_KERNEL_M4N1 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M4N1_UNIT(a_ptr, b_ptr1)

#define NEON_SGEMM_KERNEL_M1N4 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M4N1_UNIT(b_ptr1, a_ptr)

#define NEON_SGEMM_SAVE_M4N1 \
  float32x4_t ct1 = vld1q_f32(c_ptr);\
  cq01 = vfmaq_n_f32(cq01, ct1, beta);\
  vst1q_f32(c_ptr, cq01);

#define NEON_SGEMM_SAVE_M1N4_UNIT(cq01) \
  c_tmp[0] = c_tmp[0] * beta + vgetq_lane_f32(cq01, 0);\
  c_tmp[ldc] = c_tmp[ldc] * beta + vgetq_lane_f32(cq01, 1);\
  c_tmp += ldc * 2;\
  c_tmp[0] = c_tmp[0] * beta + vgetq_lane_f32(cq01, 2);\
  c_tmp[ldc] = c_tmp[ldc] * beta + vgetq_lane_f32(cq01, 3);\
  c_tmp += ldc * 2;

#define NEON_SGEMM_SAVE_M1N4 \
  float *c_tmp = c_ptr; NEON_SGEMM_SAVE_M1N4_UNIT(cq01)

#define NEON_SGEMM_KERNEL_M4N2_UNIT(a_ptr1, b_ptr1) \
  float32x4_t aq01, aq02, cq01, cq02, cq03, cq04;\
  float32x2_t bd01, bd02;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f32(0.0f);\
  uint32_t k_left = K;\
  if (k_left > 1) {\
    aq01 = vld1q_f32(a_ptr1); aq02 = vld1q_f32(a_ptr1 + 4); a_ptr1 += 8;\
    bd01 = vld1_f32(b_ptr1); bd02 = vld1_f32(b_ptr1 + 2); b_ptr1 += 4;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f32(cq02, aq01, bd01, 1);\
    aq01 = vld1q_f32(a_ptr1); bd01 = vld1_f32(b_ptr1);\
    cq03 = vfmaq_lane_f32(cq03, aq02, bd02, 0);\
    cq04 = vfmaq_lane_f32(cq04, aq02, bd02, 1);\
    aq02 = vld1q_f32(a_ptr1 + 4); a_ptr1 += 8;\
    bd02 = vld1_f32(b_ptr1 + 2); b_ptr1 += 4;\
  }\
  if (k_left > 1) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f32(cq02, aq01, bd01, 1);\
    cq03 = vfmaq_lane_f32(cq03, aq02, bd02, 0);\
    cq04 = vfmaq_lane_f32(cq04, aq02, bd02, 1); k_left -= 2;\
  }\
  cq01 = vaddq_f32(cq01, cq03);\
  cq02 = vaddq_f32(cq02, cq04);\
  if (k_left > 0) {\
    aq01 = vld1q_f32(a_ptr1); a_ptr1 += 4;\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f32(cq02, aq01, bd01, 1);\
  }

#define NEON_SGEMM_KERNEL_M4N2 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M4N2_UNIT(a_ptr, b_ptr1)

#define NEON_SGEMM_KERNEL_M2N4 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M4N2_UNIT(b_ptr1, a_ptr)

#define NEON_SGEMM_SAVE_M4N2_UNIT(cq01, cq02) \
  ct1 = vld1q_f32(c_tmp); ct2 = vld1q_f32(c_tmp + ldc);\
  cq01 = vfmaq_n_f32(cq01, ct1, beta);\
  cq02 = vfmaq_n_f32(cq02, ct2, beta);\
  vst1q_f32(c_tmp, cq01);\
  vst1q_f32(c_tmp + ldc, cq02);\
  c_tmp += ldc * 2;

#define NEON_SGEMM_SAVE_M4N2 \
  float32x4_t ct1, ct2;\
  float *c_tmp = c_ptr; NEON_SGEMM_SAVE_M4N2_UNIT(cq01, cq02)

#define NEON_SGEMM_SAVE_M2N4_UNIT(cq01, cq02) \
  ctd1 = vzipq_f32(cq01, cq02);\
  cd1 = vget_low_f32(ctd1.val[0]);\
  cd2 = vget_high_f32(ctd1.val[0]);\
  cd3 = vget_low_f32(ctd1.val[1]);\
  cd4 = vget_high_f32(ctd1.val[1]);\
  cd1 = vfma_n_f32(cd1, vld1_f32(c_tmp), beta);\
  cd2 = vfma_n_f32(cd2, vld1_f32(c_tmp + ldc), beta);\
  cd3 = vfma_n_f32(cd3, vld1_f32(c_tmp + ldc * 2), beta);\
  cd4 = vfma_n_f32(cd4, vld1_f32(c_tmp + ldc * 3), beta);\
  vst1_f32(c_tmp, cd1);\
  vst1_f32(c_tmp + ldc, cd2);\
  vst1_f32(c_tmp + ldc * 2, cd3);\
  vst1_f32(c_tmp + ldc * 3, cd4);\
  c_tmp += ldc * 4;

#define NEON_SGEMM_SAVE_M2N4 \
  float32x4x2_t ctd1; float32x2_t cd1, cd2, cd3, cd4;\
  float *c_tmp = c_ptr; NEON_SGEMM_SAVE_M2N4_UNIT(cq01, cq02)

#define NEON_SGEMM_KERNEL_M4N4 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  float32x4_t aq01, cq01, cq02, cq03, cq04;\
  float32x2_t bd01, bd02;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f32(0.0f);\
  uint32_t k_left = K;\
  if (k_left > 0) {\
    aq01 = vld1q_f32(a_ptr); a_ptr += 4;\
    bd01 = vld1_f32(b_ptr1); bd02 = vld1_f32(b_ptr1 + 2); b_ptr1 += 4;\
  }\
  for (; k_left > 1; k_left--) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f32(cq02, aq01, bd01, 1); bd01 = vld1_f32(b_ptr1);\
    cq03 = vfmaq_lane_f32(cq03, aq01, bd02, 0);\
    cq04 = vfmaq_lane_f32(cq04, aq01, bd02, 1); bd02 = vld1_f32(b_ptr1 + 2);\
    b_ptr1 += 4; aq01 = vld1q_f32(a_ptr); a_ptr += 4;\
  }\
  if (k_left > 0) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f32(cq02, aq01, bd01, 1);\
    cq03 = vfmaq_lane_f32(cq03, aq01, bd02, 0);\
    cq04 = vfmaq_lane_f32(cq04, aq01, bd02, 1);\
  }

#define NEON_SGEMM_SAVE_M4N4_UNIT(cq01, cq02, cq03, cq04) \
  ct1 = vld1q_f32(c_tmp);\
  ct2 = vld1q_f32(c_tmp + ldc);\
  ct3 = vld1q_f32(c_tmp + ldc * 2);\
  ct4 = vld1q_f32(c_tmp + ldc * 3);\
  cq01 = vfmaq_n_f32(cq01, ct1, beta);\
  cq02 = vfmaq_n_f32(cq02, ct2, beta);\
  cq03 = vfmaq_n_f32(cq03, ct3, beta);\
  cq04 = vfmaq_n_f32(cq04, ct4, beta);\
  vst1q_f32(c_tmp, cq01);\
  vst1q_f32(c_tmp + ldc, cq02);\
  vst1q_f32(c_tmp + ldc * 2, cq03);\
  vst1q_f32(c_tmp + ldc * 3, cq04); c_tmp += ldc * 4;

#define NEON_SGEMM_SAVE_M4N4 \
  float32x4_t ct1, ct2, ct3, ct4;\
  float *c_tmp = c_ptr; NEON_SGEMM_SAVE_M4N4_UNIT(cq01, cq02, cq03, cq04)

#define NEON_SGEMM_KERNEL_M8N1_UNIT(a_ptr1, b_ptr1) \
  float32x4_t aq01, aq02, aq03, aq04, cq01, cq02, cq03, cq04;\
  float32x2_t bd01;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f32(0.0f);\
  uint32_t k_left = K;\
  if (k_left > 1) {\
    aq01 = vld1q_f32(a_ptr1); aq02 = vld1q_f32(a_ptr1 + 4);\
    aq03 = vld1q_f32(a_ptr1 + 8); aq04 = vld1q_f32(a_ptr1 + 12); a_ptr1 += 16;\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0); aq01 = vld1q_f32(a_ptr1);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 0); aq02 = vld1q_f32(a_ptr1 + 4);\
    cq03 = vfmaq_lane_f32(cq03, aq03, bd01, 1); aq03 = vld1q_f32(a_ptr1 + 8);\
    cq04 = vfmaq_lane_f32(cq04, aq04, bd01, 1); aq04 = vld1q_f32(a_ptr1 + 12);\
    a_ptr1 += 16; bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  if (k_left > 1) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 0);\
    cq03 = vfmaq_lane_f32(cq03, aq03, bd01, 1);\
    cq04 = vfmaq_lane_f32(cq04, aq04, bd01, 1); k_left -= 2;\
  }\
  cq01 = vaddq_f32(cq01, cq03);\
  cq02 = vaddq_f32(cq02, cq04);\
  if (k_left > 0) {\
    float bs1 = *b_ptr1; b_ptr1++;\
    aq01 = vld1q_f32(a_ptr1);\
    aq02 = vld1q_f32(a_ptr1 + 4); a_ptr1 += 8;\
    cq01 = vfmaq_n_f32(cq01, aq01, bs1);\
    cq02 = vfmaq_n_f32(cq02, aq02, bs1);\
  }

#define NEON_SGEMM_KERNEL_M8N1 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M8N1_UNIT(a_ptr, b_ptr1)

#define NEON_SGEMM_KERNEL_M1N8 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M8N1_UNIT(b_ptr1, a_ptr)

#define NEON_SGEMM_SAVE_M8N1 \
  float32x4_t ct1, ct2;\
  ct1 = vld1q_f32(c_ptr); ct2 = vld1q_f32(c_ptr + 4);\
  cq01 = vfmaq_n_f32(cq01, ct1, beta);\
  cq02 = vfmaq_n_f32(cq02, ct2, beta);\
  vst1q_f32(c_ptr, cq01);\
  vst1q_f32(c_ptr + 4, cq02);

#define NEON_SGEMM_SAVE_M1N8 \
  float *c_tmp = c_ptr; NEON_SGEMM_SAVE_M1N4_UNIT(cq01) NEON_SGEMM_SAVE_M1N4_UNIT(cq02)

#define NEON_SGEMM_KERNEL_M8N2_UNIT(a_ptr1, b_ptr1) \
  float32x4_t aq01, aq02, cq01, cq02, cq03, cq04;\
  float32x2_t bd01;\
  cq01 = cq02 = cq03 = cq04 = vdupq_n_f32(0.0f);\
  uint32_t k_left = K;\
  if (k_left > 0) {\
    aq01 = vld1q_f32(a_ptr1); aq02 = vld1q_f32(a_ptr1 + 4); a_ptr1 += 8;\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  for (; k_left > 1; k_left--) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq03 = vfmaq_lane_f32(cq03, aq01, bd01, 1); aq01 = vld1q_f32(a_ptr1);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 0);\
    cq04 = vfmaq_lane_f32(cq04, aq02, bd01, 1); aq02 = vld1q_f32(a_ptr1 + 4);\
    a_ptr1 += 8; bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  if (k_left > 0) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq03 = vfmaq_lane_f32(cq03, aq01, bd01, 1);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 0);\
    cq04 = vfmaq_lane_f32(cq04, aq02, bd01, 1);\
  }

#define NEON_SGEMM_KERNEL_M8N2 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M8N2_UNIT(a_ptr, b_ptr1)

#define NEON_SGEMM_KERNEL_M2N8 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M8N2_UNIT(b_ptr1, a_ptr)

#define NEON_SGEMM_SAVE_M8N2_UNIT(cq01, cq02, cq03, cq04) \
  ct1 = vld1q_f32(c_tmp);\
  ct2 = vld1q_f32(c_tmp + 4);\
  ct3 = vld1q_f32(c_tmp + ldc);\
  ct4 = vld1q_f32(c_tmp + ldc + 4);\
  cq01 = vfmaq_n_f32(cq01, ct1, beta);\
  cq02 = vfmaq_n_f32(cq02, ct2, beta);\
  cq03 = vfmaq_n_f32(cq03, ct3, beta);\
  cq04 = vfmaq_n_f32(cq04, ct4, beta);\
  vst1q_f32(c_tmp, cq01);\
  vst1q_f32(c_tmp + 4, cq02);\
  vst1q_f32(c_tmp + ldc, cq03);\
  vst1q_f32(c_tmp + ldc + 4, cq04); c_tmp += 2 * ldc;

#define NEON_SGEMM_SAVE_M8N2 \
  float *c_tmp = c_ptr;\
  float32x4_t ct1, ct2, ct3, ct4;\
  NEON_SGEMM_SAVE_M8N2_UNIT(cq01, cq02, cq03, cq04)

#define NEON_SGEMM_SAVE_M2N8 \
  float32x4x2_t ctd1; float32x2_t cd1, cd2, cd3, cd4;\
  float *c_tmp = c_ptr; NEON_SGEMM_SAVE_M2N4_UNIT(cq01, cq03) NEON_SGEMM_SAVE_M2N4_UNIT(cq02, cq04)

#define NEON_SGEMM_KERNEL_M8N4_UNIT(a_ptr1, b_ptr1) \
  float32x4_t aq01, aq02, bq01, cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = cq07 = cq08 = vdupq_n_f32(0.0f);\
  uint32_t k_left = K;\
  if (k_left > 0) {\
    aq01 = vld1q_f32(a_ptr1); aq02 = vld1q_f32(a_ptr1 + 4); a_ptr1 += 8;\
    bq01 = vld1q_f32(b_ptr1); b_ptr1 += 4;\
  }\
  for (; k_left > 1; k_left--) {\
    cq01 = vfmaq_laneq_f32(cq01, aq01, bq01, 0);\
    cq03 = vfmaq_laneq_f32(cq03, aq01, bq01, 1);\
    cq05 = vfmaq_laneq_f32(cq05, aq01, bq01, 2);\
    cq07 = vfmaq_laneq_f32(cq07, aq01, bq01, 3);\
    aq01 = vld1q_f32(a_ptr1);\
    cq02 = vfmaq_laneq_f32(cq02, aq02, bq01, 0);\
    cq04 = vfmaq_laneq_f32(cq04, aq02, bq01, 1);\
    cq06 = vfmaq_laneq_f32(cq06, aq02, bq01, 2);\
    cq08 = vfmaq_laneq_f32(cq08, aq02, bq01, 3);\
    aq02 = vld1q_f32(a_ptr1 + 4); a_ptr1 += 8;\
    bq01 = vld1q_f32(b_ptr1); b_ptr1 += 4;\
  }\
  if (k_left > 0) {\
    cq01 = vfmaq_laneq_f32(cq01, aq01, bq01, 0);\
    cq03 = vfmaq_laneq_f32(cq03, aq01, bq01, 1);\
    cq05 = vfmaq_laneq_f32(cq05, aq01, bq01, 2);\
    cq07 = vfmaq_laneq_f32(cq07, aq01, bq01, 3);\
    cq02 = vfmaq_laneq_f32(cq02, aq02, bq01, 0);\
    cq04 = vfmaq_laneq_f32(cq04, aq02, bq01, 1);\
    cq06 = vfmaq_laneq_f32(cq06, aq02, bq01, 2);\
    cq08 = vfmaq_laneq_f32(cq08, aq02, bq01, 3);\
  }

#define NEON_SGEMM_KERNEL_M8N4 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M8N4_UNIT(a_ptr, b_ptr1)

#define NEON_SGEMM_KERNEL_M4N8 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  NEON_SGEMM_KERNEL_M8N4_UNIT(b_ptr1, a_ptr)

#define NEON_SGEMM_SAVE_M8N4 \
  float *c_tmp = c_ptr;\
  float32x4_t ct1, ct2, ct3, ct4;\
  NEON_SGEMM_SAVE_M8N2_UNIT(cq01, cq02, cq03, cq04)\
  NEON_SGEMM_SAVE_M8N2_UNIT(cq05, cq06, cq07, cq08)

#define TRANSPOSE_4x4(cq1, cq2, cq3, cq4) {\
  float32x4x2_t ctd1 = vzipq_f32(cq1, cq2);\
  float32x4x2_t ctd2 = vzipq_f32(cq3, cq4);\
  cq1 = vcombine_f32(vget_low_f32(ctd1.val[0]), vget_low_f32(ctd2.val[0]));\
  cq2 = vcombine_f32(vget_high_f32(ctd1.val[0]), vget_high_f32(ctd2.val[0]));\
  cq3 = vcombine_f32(vget_low_f32(ctd1.val[1]), vget_low_f32(ctd2.val[1]));\
  cq4 = vcombine_f32(vget_high_f32(ctd1.val[1]), vget_high_f32(ctd2.val[1]));\
}

#define NEON_SGEMM_SAVE_M4N8 \
  float *c_tmp = c_ptr;\
  float32x4_t ct1, ct2, ct3, ct4;\
  TRANSPOSE_4x4(cq01, cq03, cq05, cq07)\
  TRANSPOSE_4x4(cq02, cq04, cq06, cq08)\
  NEON_SGEMM_SAVE_M4N4_UNIT(cq01, cq03, cq05, cq07)\
  NEON_SGEMM_SAVE_M4N4_UNIT(cq02, cq04, cq06, cq08)

#define NEON_SGEMM_KERNEL_M8N8 \
  const float *a_ptr = a_head;\
  const float *b_ptr1 = b_head;\
  float32x4_t aq01, aq02, bq01, bq02;\
  float32x4_t cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  float32x4_t cq09, cq10, cq11, cq12, cq13, cq14, cq15, cq16;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = cq07 = cq08 = vdupq_n_f32(0.0f);\
  cq09 = cq10 = cq11 = cq12 = cq13 = cq14 = cq15 = cq16 = vdupq_n_f32(0.0f);\
  uint32_t k_left = K;\
  if (k_left > 0) {\
    aq01 = vld1q_f32(a_ptr); aq02 = vld1q_f32(a_ptr + 4); a_ptr += 8;\
    bq01 = vld1q_f32(b_ptr1); bq02 = vld1q_f32(b_ptr1 + 4); b_ptr1 += 8;\
  }\
  for (; k_left > 1; k_left--) {\
    cq01 = vfmaq_laneq_f32(cq01, aq01, bq01, 0);\
    cq03 = vfmaq_laneq_f32(cq03, aq01, bq01, 1);\
    cq05 = vfmaq_laneq_f32(cq05, aq01, bq01, 2);\
    cq07 = vfmaq_laneq_f32(cq07, aq01, bq01, 3);\
    cq02 = vfmaq_laneq_f32(cq02, aq02, bq01, 0);\
    cq04 = vfmaq_laneq_f32(cq04, aq02, bq01, 1);\
    cq06 = vfmaq_laneq_f32(cq06, aq02, bq01, 2);\
    cq08 = vfmaq_laneq_f32(cq08, aq02, bq01, 3);\
    bq01 = vld1q_f32(b_ptr1);\
    cq09 = vfmaq_laneq_f32(cq09, aq01, bq02, 0);\
    cq11 = vfmaq_laneq_f32(cq11, aq01, bq02, 1);\
    cq13 = vfmaq_laneq_f32(cq13, aq01, bq02, 2);\
    cq15 = vfmaq_laneq_f32(cq15, aq01, bq02, 3);\
    aq01 = vld1q_f32(a_ptr);\
    cq10 = vfmaq_laneq_f32(cq10, aq02, bq02, 0);\
    cq12 = vfmaq_laneq_f32(cq12, aq02, bq02, 1);\
    cq14 = vfmaq_laneq_f32(cq14, aq02, bq02, 2);\
    cq16 = vfmaq_laneq_f32(cq16, aq02, bq02, 3);\
    aq02 = vld1q_f32(a_ptr + 4); a_ptr += 8;\
    bq02 = vld1q_f32(b_ptr1 + 4); b_ptr1 += 8;\
  }\
  if (k_left > 0) {\
    cq01 = vfmaq_laneq_f32(cq01, aq01, bq01, 0);\
    cq03 = vfmaq_laneq_f32(cq03, aq01, bq01, 1);\
    cq05 = vfmaq_laneq_f32(cq05, aq01, bq01, 2);\
    cq07 = vfmaq_laneq_f32(cq07, aq01, bq01, 3);\
    cq02 = vfmaq_laneq_f32(cq02, aq02, bq01, 0);\
    cq04 = vfmaq_laneq_f32(cq04, aq02, bq01, 1);\
    cq06 = vfmaq_laneq_f32(cq06, aq02, bq01, 2);\
    cq08 = vfmaq_laneq_f32(cq08, aq02, bq01, 3);\
    cq09 = vfmaq_laneq_f32(cq09, aq01, bq02, 0);\
    cq11 = vfmaq_laneq_f32(cq11, aq01, bq02, 1);\
    cq13 = vfmaq_laneq_f32(cq13, aq01, bq02, 2);\
    cq15 = vfmaq_laneq_f32(cq15, aq01, bq02, 3);\
    cq10 = vfmaq_laneq_f32(cq10, aq02, bq02, 0);\
    cq12 = vfmaq_laneq_f32(cq12, aq02, bq02, 1);\
    cq14 = vfmaq_laneq_f32(cq14, aq02, bq02, 2);\
    cq16 = vfmaq_laneq_f32(cq16, aq02, bq02, 3);\
  }

#define NEON_SGEMM_SAVE_M8N8 \
  float *c_tmp = c_ptr;\
  float32x4_t ct1, ct2, ct3, ct4;\
  NEON_SGEMM_SAVE_M8N2_UNIT(cq01, cq02, cq03, cq04)\
  NEON_SGEMM_SAVE_M8N2_UNIT(cq05, cq06, cq07, cq08)\
  NEON_SGEMM_SAVE_M8N2_UNIT(cq09, cq10, cq11, cq12)\
  NEON_SGEMM_SAVE_M8N2_UNIT(cq13, cq14, cq15, cq16)

#define NEON_SGEMM_KERNEL_M6N1_UNIT(a_ptr1, b_ptr1) \
  uint32_t k_left = K;\
  float32x2_t cd01, cd02, cd03, cd04, cd05, cd06;\
  float32x2_t ad01, ad02, ad03, ad04, ad05, ad06, bd01;\
  cd01 = cd02 = cd03 = cd04 = cd05 = cd06 = vdup_n_f32(0.0f);\
  if (k_left > 1) {\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
    ad01 = vld1_f32(a_ptr1); ad02 = vld1_f32(a_ptr1 + 2);\
    ad03 = vld1_f32(a_ptr1 + 4); ad04 = vld1_f32(a_ptr1 + 6);\
    ad05 = vld1_f32(a_ptr1 + 8); ad06 = vld1_f32(a_ptr1 + 10);\
    a_ptr1 += 12;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cd01 = vfma_lane_f32(cd01, ad01, bd01, 0); ad01 = vld1_f32(a_ptr1);\
    cd02 = vfma_lane_f32(cd02, ad02, bd01, 0); ad02 = vld1_f32(a_ptr1 + 2);\
    cd03 = vfma_lane_f32(cd03, ad03, bd01, 0); ad03 = vld1_f32(a_ptr1 + 4);\
    cd04 = vfma_lane_f32(cd04, ad04, bd01, 1); ad04 = vld1_f32(a_ptr1 + 6);\
    cd05 = vfma_lane_f32(cd05, ad05, bd01, 1); ad05 = vld1_f32(a_ptr1 + 8);\
    cd06 = vfma_lane_f32(cd06, ad06, bd01, 1); ad06 = vld1_f32(a_ptr1 + 10);\
    a_ptr1 += 12; bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  if (k_left > 1) {\
    cd01 = vfma_lane_f32(cd01, ad01, bd01, 0);\
    cd02 = vfma_lane_f32(cd02, ad02, bd01, 0);\
    cd03 = vfma_lane_f32(cd03, ad03, bd01, 0);\
    cd04 = vfma_lane_f32(cd04, ad04, bd01, 1);\
    cd05 = vfma_lane_f32(cd05, ad05, bd01, 1);\
    cd06 = vfma_lane_f32(cd06, ad06, bd01, 1); k_left -= 2;\
  }\
  cd01 = vadd_f32(cd01, cd04);\
  cd02 = vadd_f32(cd02, cd05);\
  cd03 = vadd_f32(cd03, cd06);\
  if (k_left > 0) {\
    float bs1 = *b_ptr1; b_ptr1++;\
    ad01 = vld1_f32(a_ptr1);\
    ad02 = vld1_f32(a_ptr1 + 2);\
    ad03 = vld1_f32(a_ptr1 + 4); a_ptr1 += 6;\
    cd01 = vfma_n_f32(cd01, ad01, bs1);\
    cd02 = vfma_n_f32(cd02, ad02, bs1);\
    cd03 = vfma_n_f32(cd03, ad03, bs1);\
  }

#define NEON_SGEMM_KERNEL_M6N1 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M6N1_UNIT(a_ptr, b_ptr)

#define NEON_SGEMM_KERNEL_M1N6 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M6N1_UNIT(b_ptr, a_ptr)

#define NEON_SGEMM_SAVE_M6N1 \
  float32x2_t ct1, ct2, ct3;\
  ct1 = vld1_f32(c_ptr); ct2 = vld1_f32(c_ptr + 2); ct3 = vld1_f32(c_ptr + 4);\
  cd01 = vfma_n_f32(cd01, ct1, beta);\
  cd02 = vfma_n_f32(cd02, ct2, beta);\
  cd03 = vfma_n_f32(cd03, ct3, beta);\
  vst1_f32(c_ptr, cd01); vst1_f32(c_ptr + 2, cd02); vst1_f32(c_ptr + 4, cd03);

#define NEON_SGEMM_SAVE_M1N6 \
  float *c_tmp = c_ptr;\
  NEON_SGEMM_SAVE_M1N2_UNIT(cd01) NEON_SGEMM_SAVE_M1N2_UNIT(cd02) NEON_SGEMM_SAVE_M1N2_UNIT(cd03)

#define NEON_SGEMM_KERNEL_M6N2_UNIT(a_ptr1, b_ptr1) \
  uint32_t k_left = K;\
  float32x2_t cd01, cd02, cd03, cd04, cd05, cd06;\
  float32x2_t ad01, ad02, ad03, bd01;\
  cd01 = cd02 = cd03 = cd04 = cd05 = cd06 = vdup_n_f32(0.0f);\
  if (k_left > 0) {\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
    ad01 = vld1_f32(a_ptr1); ad02 = vld1_f32(a_ptr1 + 2);\
    ad03 = vld1_f32(a_ptr1 + 4); a_ptr1 += 6;\
  }\
  for (; k_left > 1; k_left--) {\
    cd01 = vfma_lane_f32(cd01, ad01, bd01, 0);\
    cd04 = vfma_lane_f32(cd04, ad01, bd01, 1); ad01 = vld1_f32(a_ptr1);\
    cd02 = vfma_lane_f32(cd02, ad02, bd01, 0);\
    cd05 = vfma_lane_f32(cd05, ad02, bd01, 1); ad02 = vld1_f32(a_ptr1 + 2);\
    cd03 = vfma_lane_f32(cd03, ad03, bd01, 0);\
    cd06 = vfma_lane_f32(cd06, ad03, bd01, 1); ad03 = vld1_f32(a_ptr1 + 4);\
    a_ptr1 += 6; bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  if (k_left > 0) {\
    cd01 = vfma_lane_f32(cd01, ad01, bd01, 0);\
    cd04 = vfma_lane_f32(cd04, ad01, bd01, 1);\
    cd02 = vfma_lane_f32(cd02, ad02, bd01, 0);\
    cd05 = vfma_lane_f32(cd05, ad02, bd01, 1);\
    cd03 = vfma_lane_f32(cd03, ad03, bd01, 0);\
    cd06 = vfma_lane_f32(cd06, ad03, bd01, 1);\
  }

#define NEON_SGEMM_KERNEL_M6N2 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M6N2_UNIT(a_ptr, b_ptr)

#define NEON_SGEMM_KERNEL_M2N6 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M6N2_UNIT(b_ptr, a_ptr)

#define TRANS_M2N2(cd01, cd02) \
  cdd1 = vzip_f32(cd01, cd02); cd01 = cdd1.val[0]; cd02 = cdd1.val[1];

#define NEON_SGEMM_SAVE_M6N2 \
  float32x2_t ct1, ct2; float *c_tmp = c_ptr;\
  NEON_SGEMM_SAVE_M2N2_UNIT(cd01, cd04) c_tmp = c_ptr + 2;\
  NEON_SGEMM_SAVE_M2N2_UNIT(cd02, cd05) c_tmp = c_ptr + 4;\
  NEON_SGEMM_SAVE_M2N2_UNIT(cd03, cd06)

#define NEON_SGEMM_SAVE_M2N6 \
  float32x2x2_t cdd1; float32x2_t ct1, ct2; float *c_tmp = c_ptr;\
  TRANS_M2N2(cd01, cd04) TRANS_M2N2(cd02, cd05) TRANS_M2N2(cd03, cd06)\
  NEON_SGEMM_SAVE_M2N2_UNIT(cd01, cd04)\
  NEON_SGEMM_SAVE_M2N2_UNIT(cd02, cd05)\
  NEON_SGEMM_SAVE_M2N2_UNIT(cd03, cd06)

#define NEON_SGEMM_KERNEL_M6N4_UNIT(a_ptr1, b_ptr1) \
  uint32_t k_left = K;\
  float32x4_t cq01, cq02, cq03, cq04, cq05, cq06;\
  float32x4_t bq01; float32x2_t ad01, ad02, ad03;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = vdupq_n_f32(0.0f);\
  if (k_left > 0) {\
    bq01 = vld1q_f32(b_ptr1); b_ptr1 += 4;\
    ad01 = vld1_f32(a_ptr1); ad02 = vld1_f32(a_ptr1 + 2);\
    ad03 = vld1_f32(a_ptr1 + 4); a_ptr1 += 6;\
  }\
  for (; k_left > 1; k_left--) {\
    cq01 = vfmaq_lane_f32(cq01, bq01, ad01, 0);\
    cq02 = vfmaq_lane_f32(cq02, bq01, ad01, 1); ad01 = vld1_f32(a_ptr1);\
    cq03 = vfmaq_lane_f32(cq03, bq01, ad02, 0);\
    cq04 = vfmaq_lane_f32(cq04, bq01, ad02, 1); ad02 = vld1_f32(a_ptr1 + 2);\
    cq05 = vfmaq_lane_f32(cq05, bq01, ad03, 0);\
    cq06 = vfmaq_lane_f32(cq06, bq01, ad03, 1); ad03 = vld1_f32(a_ptr1 + 4);\
    a_ptr1 += 6; bq01 = vld1q_f32(b_ptr1); b_ptr1 += 4;\
  }\
  if (k_left > 0) {\
    cq01 = vfmaq_lane_f32(cq01, bq01, ad01, 0);\
    cq02 = vfmaq_lane_f32(cq02, bq01, ad01, 1);\
    cq03 = vfmaq_lane_f32(cq03, bq01, ad02, 0);\
    cq04 = vfmaq_lane_f32(cq04, bq01, ad02, 1);\
    cq05 = vfmaq_lane_f32(cq05, bq01, ad03, 0);\
    cq06 = vfmaq_lane_f32(cq06, bq01, ad03, 1);\
  }

#define NEON_SGEMM_KERNEL_M6N4 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M6N4_UNIT(a_ptr, b_ptr)

#define NEON_SGEMM_KERNEL_M4N6 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M6N4_UNIT(b_ptr, a_ptr)

#define NEON_SGEMM_SAVE_M6N4 \
  float32x4x2_t ctd1; float32x2_t cd1, cd2, cd3, cd4; float *c_tmp = c_ptr;\
  NEON_SGEMM_SAVE_M2N4_UNIT(cq01, cq02) c_tmp = c_ptr + 2;\
  NEON_SGEMM_SAVE_M2N4_UNIT(cq03, cq04) c_tmp = c_ptr + 4;\
  NEON_SGEMM_SAVE_M2N4_UNIT(cq05, cq06)

#define NEON_SGEMM_SAVE_M4N6 \
  float32x4_t ct1, ct2; float *c_tmp = c_ptr;\
  NEON_SGEMM_SAVE_M4N2_UNIT(cq01, cq02) NEON_SGEMM_SAVE_M4N2_UNIT(cq03, cq04)\
  NEON_SGEMM_SAVE_M4N2_UNIT(cq05, cq06)

#define NEON_SGEMM_KERNEL_M12N1_UNIT(a_ptr1, b_ptr1) \
  uint32_t k_left = K;\
  float32x4_t cq01, cq02, cq03, cq04, cq05, cq06;\
  float32x4_t aq01, aq02, aq03, aq04, aq05, aq06;\
  float32x2_t bd01;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = vdupq_n_f32(0.0f);\
  if (k_left > 1) {\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
    aq01 = vld1q_f32(a_ptr1); aq02 = vld1q_f32(a_ptr1 + 4);\
    aq03 = vld1q_f32(a_ptr1 + 8); aq04 = vld1q_f32(a_ptr1 + 12);\
    aq05 = vld1q_f32(a_ptr1 + 16); aq06 = vld1q_f32(a_ptr1 + 20);\
    a_ptr1 += 24;\
  }\
  for (; k_left > 3; k_left -= 2) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0); aq01 = vld1q_f32(a_ptr1);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 0); aq02 = vld1q_f32(a_ptr1 + 4);\
    cq03 = vfmaq_lane_f32(cq03, aq03, bd01, 0); aq03 = vld1q_f32(a_ptr1 + 8);\
    cq04 = vfmaq_lane_f32(cq04, aq04, bd01, 1); aq04 = vld1q_f32(a_ptr1 + 12);\
    cq05 = vfmaq_lane_f32(cq05, aq05, bd01, 1); aq05 = vld1q_f32(a_ptr1 + 16);\
    cq06 = vfmaq_lane_f32(cq06, aq06, bd01, 1); aq06 = vld1q_f32(a_ptr1 + 20);\
    a_ptr1 += 24; bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  if (k_left > 1) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 0);\
    cq03 = vfmaq_lane_f32(cq03, aq03, bd01, 0);\
    cq04 = vfmaq_lane_f32(cq04, aq04, bd01, 1);\
    cq05 = vfmaq_lane_f32(cq05, aq05, bd01, 1);\
    cq06 = vfmaq_lane_f32(cq06, aq06, bd01, 1); k_left -= 2;\
  }\
  cq01 = vaddq_f32(cq01, cq04);\
  cq02 = vaddq_f32(cq02, cq05);\
  cq03 = vaddq_f32(cq03, cq06);\
  if (k_left > 0) {\
    float bs1 = *b_ptr1; b_ptr1++;\
    aq01 = vld1q_f32(a_ptr1); aq02 = vld1q_f32(a_ptr1 + 4);\
    aq03 = vld1q_f32(a_ptr1 + 8); a_ptr1 += 12;\
    cq01 = vfmaq_n_f32(cq01, aq01, bs1);\
    cq02 = vfmaq_n_f32(cq02, aq02, bs1);\
    cq03 = vfmaq_n_f32(cq03, aq03, bs1);\
  }

#define NEON_SGEMM_KERNEL_M12N1 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M12N1_UNIT(a_ptr, b_ptr)

#define NEON_SGEMM_KERNEL_M1N12 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M12N1_UNIT(b_ptr, a_ptr)

#define NEON_SGEMM_SAVE_M12N1 \
  float32x4_t ct1, ct2, ct3;\
  ct1 = vld1q_f32(c_ptr); ct2 = vld1q_f32(c_ptr + 4); ct3 = vld1q_f32(c_ptr + 8);\
  cq01 = vfmaq_n_f32(cq01, ct1, beta);\
  cq02 = vfmaq_n_f32(cq02, ct2, beta);\
  cq03 = vfmaq_n_f32(cq03, ct3, beta);\
  vst1q_f32(c_ptr, cq01); vst1q_f32(c_ptr + 4, cq02); vst1q_f32(c_ptr + 8, cq03);

#define NEON_SGEMM_SAVE_M1N12 \
  float *c_tmp = c_ptr;\
  NEON_SGEMM_SAVE_M1N4_UNIT(cq01) NEON_SGEMM_SAVE_M1N4_UNIT(cq02) NEON_SGEMM_SAVE_M1N4_UNIT(cq03)

#define NEON_SGEMM_KERNEL_M12N2_UNIT(a_ptr1, b_ptr1) \
  uint32_t k_left = K;\
  float32x4_t cq01, cq02, cq03, cq04, cq05, cq06;\
  float32x4_t aq01, aq02, aq03; float32x2_t bd01;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = vdupq_n_f32(0.0f);\
  if (k_left > 0) {\
    bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
    aq01 = vld1q_f32(a_ptr1); aq02 = vld1q_f32(a_ptr1 + 4);\
    aq03 = vld1q_f32(a_ptr1 + 8); a_ptr1 += 12;\
  }\
  for (; k_left > 1; k_left--) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq04 = vfmaq_lane_f32(cq04, aq01, bd01, 1); aq01 = vld1q_f32(a_ptr1);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 0);\
    cq05 = vfmaq_lane_f32(cq05, aq02, bd01, 1); aq02 = vld1q_f32(a_ptr1 + 4);\
    cq03 = vfmaq_lane_f32(cq03, aq03, bd01, 0);\
    cq06 = vfmaq_lane_f32(cq06, aq03, bd01, 1); aq03 = vld1q_f32(a_ptr1 + 8);\
    a_ptr1 += 12; bd01 = vld1_f32(b_ptr1); b_ptr1 += 2;\
  }\
  if (k_left > 0) {\
    cq01 = vfmaq_lane_f32(cq01, aq01, bd01, 0);\
    cq04 = vfmaq_lane_f32(cq04, aq01, bd01, 1);\
    cq02 = vfmaq_lane_f32(cq02, aq02, bd01, 0);\
    cq05 = vfmaq_lane_f32(cq05, aq02, bd01, 1);\
    cq03 = vfmaq_lane_f32(cq03, aq03, bd01, 0);\
    cq06 = vfmaq_lane_f32(cq06, aq03, bd01, 1);\
  }

#define NEON_SGEMM_KERNEL_M12N2 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M12N2_UNIT(a_ptr, b_ptr)

#define NEON_SGEMM_KERNEL_M2N12 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M12N2_UNIT(b_ptr, a_ptr)

#define NEON_SGEMM_SAVE_M12N2 \
  float32x4_t ct1, ct2; float *c_tmp = c_ptr;\
  NEON_SGEMM_SAVE_M4N2_UNIT(cq01, cq04) c_tmp = c_ptr + 4;\
  NEON_SGEMM_SAVE_M4N2_UNIT(cq02, cq05) c_tmp = c_ptr + 8;\
  NEON_SGEMM_SAVE_M4N2_UNIT(cq03, cq06)

#define NEON_SGEMM_SAVE_M2N12 \
  float32x4x2_t ctd1; float32x2_t cd1, cd2, cd3, cd4;\
  float *c_tmp = c_ptr; NEON_SGEMM_SAVE_M2N4_UNIT(cq01, cq04)\
  NEON_SGEMM_SAVE_M2N4_UNIT(cq02, cq05) NEON_SGEMM_SAVE_M2N4_UNIT(cq03, cq06)

#define NEON_SGEMM_KERNEL_M12N4_UNIT(a_ptr1, b_ptr1) \
  uint32_t k_left = K;\
  float32x4_t cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  float32x4_t cq09, cq10, cq11, cq12, aq01, aq02, aq03, bq01;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = vdupq_n_f32(0.0f);\
  cq07 = cq08 = cq09 = cq10 = cq11 = cq12 = vdupq_n_f32(0.0f);\
  if (k_left > 0) {\
    bq01 = vld1q_f32(b_ptr1); b_ptr1 += 4;\
    aq01 = vld1q_f32(a_ptr1); aq02 = vld1q_f32(a_ptr1 + 4);\
    aq03 = vld1q_f32(a_ptr1 + 8); a_ptr1 += 12;\
  }\
  for (; k_left > 1; k_left--) {\
    cq01 = vfmaq_laneq_f32(cq01, aq01, bq01, 0);\
    cq04 = vfmaq_laneq_f32(cq04, aq01, bq01, 1);\
    cq07 = vfmaq_laneq_f32(cq07, aq01, bq01, 2);\
    cq10 = vfmaq_laneq_f32(cq10, aq01, bq01, 3);\
    aq01 = vld1q_f32(a_ptr1);\
    cq02 = vfmaq_laneq_f32(cq02, aq02, bq01, 0);\
    cq05 = vfmaq_laneq_f32(cq05, aq02, bq01, 1);\
    cq08 = vfmaq_laneq_f32(cq08, aq02, bq01, 2);\
    cq11 = vfmaq_laneq_f32(cq11, aq02, bq01, 3);\
    aq02 = vld1q_f32(a_ptr1 + 4);\
    cq03 = vfmaq_laneq_f32(cq03, aq03, bq01, 0);\
    cq06 = vfmaq_laneq_f32(cq06, aq03, bq01, 1);\
    cq09 = vfmaq_laneq_f32(cq09, aq03, bq01, 2);\
    cq12 = vfmaq_laneq_f32(cq12, aq03, bq01, 3);\
    aq03 = vld1q_f32(a_ptr1 + 8); a_ptr1 += 12;\
    bq01 = vld1q_f32(b_ptr1); b_ptr1 += 4;\
  }\
  if (k_left > 0) {\
    cq01 = vfmaq_laneq_f32(cq01, aq01, bq01, 0);\
    cq04 = vfmaq_laneq_f32(cq04, aq01, bq01, 1);\
    cq07 = vfmaq_laneq_f32(cq07, aq01, bq01, 2);\
    cq10 = vfmaq_laneq_f32(cq10, aq01, bq01, 3);\
    cq02 = vfmaq_laneq_f32(cq02, aq02, bq01, 0);\
    cq05 = vfmaq_laneq_f32(cq05, aq02, bq01, 1);\
    cq08 = vfmaq_laneq_f32(cq08, aq02, bq01, 2);\
    cq11 = vfmaq_laneq_f32(cq11, aq02, bq01, 3);\
    cq03 = vfmaq_laneq_f32(cq03, aq03, bq01, 0);\
    cq06 = vfmaq_laneq_f32(cq06, aq03, bq01, 1);\
    cq09 = vfmaq_laneq_f32(cq09, aq03, bq01, 2);\
    cq12 = vfmaq_laneq_f32(cq12, aq03, bq01, 3);\
  }

#define NEON_SGEMM_KERNEL_M12N4 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M12N4_UNIT(a_ptr, b_ptr)

#define NEON_SGEMM_KERNEL_M4N12 \
  const float *b_ptr = b_head;\
  const float *a_ptr = a_head;\
  NEON_SGEMM_KERNEL_M12N4_UNIT(b_ptr, a_ptr)

#define NEON_SGEMM_SAVE_M12N4 \
  float32x4_t ct1, ct2, ct3, ct4;\
  float *c_tmp = c_ptr; NEON_SGEMM_SAVE_M4N4_UNIT(cq01, cq04, cq07, cq10)\
  c_tmp = c_ptr + 4; NEON_SGEMM_SAVE_M4N4_UNIT(cq02, cq05, cq08, cq11)\
  c_tmp = c_ptr + 8; NEON_SGEMM_SAVE_M4N4_UNIT(cq03, cq06, cq09, cq12)

#define NEON_SGEMM_SAVE_M4N12 \
  float *c_tmp = c_ptr;\
  float32x4_t ct1, ct2, ct3, ct4;\
  TRANSPOSE_4x4(cq01, cq04, cq07, cq10)\
  TRANSPOSE_4x4(cq02, cq05, cq08, cq11)\
  TRANSPOSE_4x4(cq03, cq06, cq09, cq12)\
  NEON_SGEMM_SAVE_M4N4_UNIT(cq01, cq04, cq07, cq10)\
  NEON_SGEMM_SAVE_M4N4_UNIT(cq02, cq05, cq08, cq11)\
  NEON_SGEMM_SAVE_M4N4_UNIT(cq03, cq06, cq09, cq12)

#define NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(mdim, ndim) \
static inline void inline_dualpack_gemm_afloat_bfloat_cfloat_m##mdim##_n##ndim(\
  const float *a_head, const float *b_head, float *c_ptr,\
  uint32_t K, float beta, uint32_t ldc) {\
  NEON_SGEMM_KERNEL_M##mdim##N##ndim\
  NEON_SGEMM_SAVE_M##mdim##N##ndim\
}

NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 1)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 2)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 1)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 2)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 4)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 4)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 1)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 2)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 4)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 8)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 8)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 8)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 1)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 2)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 4)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 8)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 6)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 6)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 6)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(6, 1)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(6, 2)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(6, 4)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 12)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 12)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 12)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 1)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 2)
NEON_SGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 4)

#endif

