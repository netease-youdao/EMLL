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


#include "arm_neon/NeonIntOpSign.h"

#ifndef INCLUDE_I8I32DOT_GEMM_KERNEL
#define INCLUDE_I8I32DOT_GEMM_KERNEL

static inline void pref_c(I32 *dat) {
  __asm__ ("prfm pstl1keep,[%0]\n\t"::"r"(dat):);
}

#define PREF_N1 pref_c(c_pref); c_pref += ldc;
#define PREF_N2 PREF_N1 PREF_N1
#define PREF_N4 PREF_N2 PREF_N2
#define PREF_N8 PREF_N4 PREF_N4
#define PREF_N12 PREF_N8 PREF_N4

/* NOTE that the K actually means k/4 IN THIS FILE */

/* unaligned load of 4 8-bit int to a S register */
#define UNALIGNED_LD4B_SREG(var, ptr) \
  __asm__("ldr %s0,[%1]\n\t":"=w"(var):"r"(ptr):"memory")

#define VLD1(ptr) VREINTERPRET_I8_I32(VLD1_I32(ptr))

#define VLD1Q(ptr) VREINTERPRETQ_I8_I32(VLD1Q_I32(ptr))

#define NORMAL_KERNEL_SETUP(a_head, b_head) \
  uint32_t kdiv4_left = K;\
  const I32 *a_rd = a_head;\
  const I32 *b_rd = b_head;

#define KERNEL_M1N1 \
  I32X4 cq1, cq2;\
  cq1 = cq2 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, aq2, bq1, bq2;\
  if (kdiv4_left > 3) {\
    aq1 = VLD1Q(a_rd); a_rd += 4;\
    bq1 = VLD1Q(b_rd); b_rd += 4;\
  }\
  for (; kdiv4_left > 11; kdiv4_left -= 8) {\
    aq2 = VLD1Q(a_rd);\
    bq2 = VLD1Q(b_rd);\
    cq1 = VDOTQ_I32(cq1, aq1, bq1);\
    aq1 = VLD1Q(a_rd + 4); a_rd += 8;\
    bq1 = VLD1Q(b_rd + 4); b_rd += 8;\
    cq2 = VDOTQ_I32(cq2, aq2, bq2);\
  }\
  if (kdiv4_left > 7) {\
    aq2 = VLD1Q(a_rd); a_rd += 4;\
    bq2 = VLD1Q(b_rd); b_rd += 4;\
    cq1 = VDOTQ_I32(cq1, aq1, bq1);\
    cq2 = VDOTQ_I32(cq2, aq2, bq2);\
    kdiv4_left -= 8;\
  } else if (kdiv4_left > 3) {\
    cq1 = VDOTQ_I32(cq1, aq1, bq1);\
    kdiv4_left -= 4;\
  }\
  cq1 = VADDQ_I32(cq1, cq2);\
  I32X2 cd1 = VADD_I32(VGET_LOW_I32(cq1), VGET_HIGH_I32(cq1));\
  if (kdiv4_left > 1) {\
    I8X8 ad1 = VLD1(a_rd); a_rd += 2;\
    I8X8 bd1 = VLD1(b_rd); b_rd += 2;\
    cd1 = VDOT_I32(cd1, ad1, bd1);\
    kdiv4_left -= 2;\
  }\
  if (kdiv4_left > 0) {\
    I8X8 ad1, bd1;\
    UNALIGNED_LD4B_SREG(ad1, a_rd); a_rd++;\
    UNALIGNED_LD4B_SREG(bd1, b_rd); b_rd++;\
    cd1 = VDOT_I32(cd1, ad1, bd1);\
  }\
  I32 cs1 = VGET_LANE_I32(cd1, 0) + VGET_LANE_I32(cd1, 1);

#define SAVE_M1N1 *c_ptr = c_ptr[0] * beta + cs1;

#define KERNEL_M2N1_UNIT(a_head, b_head) \
  I32X2 cd1, cd2;\
  cd1 = cd2 = VDUP_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X8 ad1, ad2, bd1;\
  if (kdiv4_left > 1) {\
    ad1 = VLD1(a_rd); ad2 = VLD1(a_rd + 2); a_rd += 4;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  for (; kdiv4_left > 3; kdiv4_left -= 2) {\
    cd1 = VDOT_LANE_I32(cd1, ad1, bd1, 0); ad1 = VLD1(a_rd);\
    cd2 = VDOT_LANE_I32(cd2, ad2, bd1, 1); ad2 = VLD1(a_rd + 2);\
    a_rd += 4; bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  if (kdiv4_left > 1) {\
    cd1 = VDOT_LANE_I32(cd1, ad1, bd1, 0);\
    cd2 = VDOT_LANE_I32(cd2, ad2, bd1, 1);\
    kdiv4_left -= 2;\
  }\
  cd1 = VADD_I32(cd1, cd2);\
  if (kdiv4_left > 0) {\
    UNALIGNED_LD4B_SREG(bd1, b_rd); b_rd++;\
    ad1 = VLD1(a_rd); a_rd += 2;\
    cd1 = VDOT_LANE_I32(cd1, ad1, bd1, 0);\
  }

#define KERNEL_M2N1 KERNEL_M2N1_UNIT(a_head, b_head)
#define KERNEL_M1N2 KERNEL_M2N1_UNIT(b_head, a_head)

#define SAVE_M2N1 \
  cd1 = VMLA_N_I32(cd1, VLD1_I32(c_ptr), beta);\
  VST1_I32(c_ptr, cd1);

#define SAVE_M1N2 \
  c_ptr[0] = c_ptr[0] * beta + VGET_LANE_I32(cd1, 0);\
  c_ptr[ldc] = c_ptr[ldc] * beta + VGET_LANE_I32(cd1, 1);

#define KERNEL_M2N2 \
  I32X2 cd1, cd2;\
  cd1 = cd2 = VDUP_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X8 ad1, bd1;\
  if (kdiv4_left > 0) {\
    ad1 = VLD1(a_rd); a_rd += 2;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  for (; kdiv4_left > 1; kdiv4_left--) {\
    cd1 = VDOT_LANE_I32(cd1, ad1, bd1, 0);\
    cd2 = VDOT_LANE_I32(cd2, ad1, bd1, 1);\
    ad1 = VLD1(a_rd); a_rd += 2;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  if (kdiv4_left > 0) {\
    cd1 = VDOT_LANE_I32(cd1, ad1, bd1, 0);\
    cd2 = VDOT_LANE_I32(cd2, ad1, bd1, 1);\
  }

#define SAVE_M2N2 \
  cd1 = VMLA_N_I32(cd1, VLD1_I32(c_ptr), beta);\
  cd2 = VMLA_N_I32(cd2, VLD1_I32(c_ptr + ldc), beta);\
  VST1_I32(c_ptr, cd1); VST1_I32(c_ptr + ldc, cd2);

#define KERNEL_M4N1_UNIT(a_head, b_head) \
  I32X4 cq1, cq2;\
  cq1 = cq2 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, aq2;\
  I8X8 bd1;\
  if (kdiv4_left > 1) {\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4); a_rd += 8;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  for (; kdiv4_left > 3; kdiv4_left -= 2) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0); aq1 = VLD1Q(a_rd);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 1); aq2 = VLD1Q(a_rd + 4);\
    a_rd += 8; bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  if (kdiv4_left > 1) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 1);\
    kdiv4_left -= 2;\
  }\
  cq1 = VADDQ_I32(cq1, cq2);\
  if (kdiv4_left > 0) {\
    UNALIGNED_LD4B_SREG(bd1, b_rd); b_rd++;\
    aq1 = VLD1Q(a_rd); a_rd += 4;\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
  }

#define KERNEL_M4N1 KERNEL_M4N1_UNIT(a_head, b_head)
#define KERNEL_M1N4 KERNEL_M4N1_UNIT(b_head, a_head)

#define SAVE_M4N1 \
  cq1 = VMLAQ_N_I32(cq1, VLD1Q_I32(c_ptr), beta);\
  VST1Q_I32(c_ptr, cq1);

#define UNIT_SAVE_M1N4(cq1) \
  c_tmp[0] = c_tmp[0] * beta + VGETQ_LANE_I32(cq1, 0);\
  c_tmp[ldc] = c_tmp[ldc] * beta + VGETQ_LANE_I32(cq1, 1);\
  c_tmp += ldc * 2;\
  c_tmp[0] = c_tmp[0] * beta + VGETQ_LANE_I32(cq1, 2);\
  c_tmp[ldc] = c_tmp[ldc] * beta + VGETQ_LANE_I32(cq1, 3);\
  c_tmp += ldc * 2;

#define SAVE_M1N4 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M1N4(cq1)

#define KERNEL_M4N2_UNIT(a_head, b_head) \
  I32X4 cq1, cq2;\
  cq1 = cq2 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1; I8X8 bd1;\
  if (kdiv4_left > 0) {\
    aq1 = VLD1Q(a_rd); a_rd += 4;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  for (; kdiv4_left > 1; kdiv4_left--) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq2 = VDOTQ_LANE_I32(cq2, aq1, bd1, 1);\
    aq1 = VLD1Q(a_rd); a_rd += 4;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  if (kdiv4_left > 0) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq2 = VDOTQ_LANE_I32(cq2, aq1, bd1, 1);\
  }

#define KERNEL_M4N2 KERNEL_M4N2_UNIT(a_head, b_head)
#define KERNEL_M2N4 KERNEL_M4N2_UNIT(b_head, a_head)

#define UNIT_SAVE_M4N2(cq1, cq2) \
  cq1 = VMLAQ_N_I32(cq1, VLD1Q_I32(c_tmp), beta);\
  cq2 = VMLAQ_N_I32(cq2, VLD1Q_I32(c_tmp + ldc), beta);\
  VST1Q_I32(c_tmp, cq1); VST1Q_I32(c_tmp + ldc, cq2);\
  c_tmp += ldc * 2;

#define SAVE_M4N2 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M4N2(cq1, cq2)

#define UNIT_SAVE_M2N4(cq1, cq2) {\
  I32X4 t1 = VZIP1Q_I32(cq1, cq2);\
  I32X4 t2 = VZIP2Q_I32(cq1, cq2);\
  I32X2 l1 = VMLA_N_I32(VGET_LOW_I32(t1), VLD1_I32(c_tmp), beta);\
  I32X2 l2 = VMLA_N_I32(VGET_HIGH_I32(t1), VLD1_I32(c_tmp + ldc), beta);\
  VST1_I32(c_tmp, l1); VST1_I32(c_tmp + ldc, l2); c_tmp += ldc * 2;\
  I32X2 l3 = VMLA_N_I32(VGET_LOW_I32(t2), VLD1_I32(c_tmp), beta);\
  I32X2 l4 = VMLA_N_I32(VGET_HIGH_I32(t2), VLD1_I32(c_tmp + ldc), beta);\
  VST1_I32(c_tmp, l3); VST1_I32(c_tmp + ldc, l4); c_tmp += ldc * 2;\
}

#define SAVE_M2N4 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M2N4(cq1, cq2)

#define KERNEL_M4N4 \
  I32X4 cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, bq1;\
  if (kdiv4_left > 0) {\
    aq1 = VLD1Q(a_rd); a_rd += 4;\
    bq1 = VLD1Q(b_rd); b_rd += 4;\
  }\
  for (; kdiv4_left > 1; kdiv4_left--) {\
    cq1 = VDOTQ_LANEQ_I32(cq1, aq1, bq1, 0);\
    cq2 = VDOTQ_LANEQ_I32(cq2, aq1, bq1, 1);\
    cq3 = VDOTQ_LANEQ_I32(cq3, aq1, bq1, 2);\
    cq4 = VDOTQ_LANEQ_I32(cq4, aq1, bq1, 3);\
    aq1 = VLD1Q(a_rd); a_rd += 4;\
    bq1 = VLD1Q(b_rd); b_rd += 4;\
  }\
  if (kdiv4_left > 0) {\
    cq1 = VDOTQ_LANEQ_I32(cq1, aq1, bq1, 0);\
    cq2 = VDOTQ_LANEQ_I32(cq2, aq1, bq1, 1);\
    cq3 = VDOTQ_LANEQ_I32(cq3, aq1, bq1, 2);\
    cq4 = VDOTQ_LANEQ_I32(cq4, aq1, bq1, 3);\
  }

#define SAVE_M4N4 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M4N2(cq1, cq2) UNIT_SAVE_M4N2(cq3, cq4)

#define KERNEL_M8N1_UNIT(a_head, b_head) \
  I32X4 cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, aq2, aq3, aq4;\
  I8X8 bd1;\
  if (kdiv4_left > 1) {\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4);\
    aq3 = VLD1Q(a_rd + 8); aq4 = VLD1Q(a_rd + 12); a_rd += 16;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  for (; kdiv4_left > 3; kdiv4_left -= 2) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0); aq1 = VLD1Q(a_rd);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0); aq2 = VLD1Q(a_rd + 4);\
    cq3 = VDOTQ_LANE_I32(cq3, aq3, bd1, 1); aq3 = VLD1Q(a_rd + 8);\
    cq4 = VDOTQ_LANE_I32(cq4, aq4, bd1, 1); aq4 = VLD1Q(a_rd + 12);\
    a_rd += 16; bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  if (kdiv4_left > 1) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0);\
    cq3 = VDOTQ_LANE_I32(cq3, aq3, bd1, 1);\
    cq4 = VDOTQ_LANE_I32(cq4, aq4, bd1, 1);\
    kdiv4_left -= 2;\
  }\
  cq1 = VADDQ_I32(cq1, cq3); cq2 = VADDQ_I32(cq2, cq4);\
  if (kdiv4_left > 0) {\
    UNALIGNED_LD4B_SREG(bd1, b_rd); b_rd++;\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4); a_rd += 8;\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0);\
  }

#define KERNEL_M8N1 KERNEL_M8N1_UNIT(a_head, b_head)
#define KERNEL_M1N8 KERNEL_M8N1_UNIT(b_head, a_head)

#define SAVE_M8N1 \
  cq1 = VMLAQ_N_I32(cq1, VLD1Q_I32(c_ptr), beta);\
  cq2 = VMLAQ_N_I32(cq2, VLD1Q_I32(c_ptr + 4), beta);\
  VST1Q_I32(c_ptr, cq1); VST1Q_I32(c_ptr + 4, cq2);

#define SAVE_M1N8 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M1N4(cq1) UNIT_SAVE_M1N4(cq2)

#define KERNEL_M8N2_UNIT(a_head, b_head) \
  I32X4 cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, aq2;\
  I8X8 bd1;\
  if (kdiv4_left > 0) {\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4); a_rd += 8;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  for (; kdiv4_left > 1; kdiv4_left--) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq3 = VDOTQ_LANE_I32(cq3, aq1, bd1, 1);\
    aq1 = VLD1Q(a_rd);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0);\
    cq4 = VDOTQ_LANE_I32(cq4, aq2, bd1, 1);\
    aq2 = VLD1Q(a_rd + 4); a_rd += 8;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  if (kdiv4_left > 0) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq3 = VDOTQ_LANE_I32(cq3, aq1, bd1, 1);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0);\
    cq4 = VDOTQ_LANE_I32(cq4, aq2, bd1, 1);\
  }

#define KERNEL_M8N2 KERNEL_M8N2_UNIT(a_head, b_head)
#define KERNEL_M2N8 KERNEL_M8N2_UNIT(b_head, a_head)

#define UNIT_SAVE_M8N2(cq1, cq2, cq3, cq4) \
  cq1 = VMLAQ_N_I32(cq1, VLD1Q_I32(c_tmp), beta);\
  cq2 = VMLAQ_N_I32(cq2, VLD1Q_I32(c_tmp + 4), beta);\
  cq3 = VMLAQ_N_I32(cq3, VLD1Q_I32(c_tmp + ldc), beta);\
  cq4 = VMLAQ_N_I32(cq4, VLD1Q_I32(c_tmp + ldc + 4), beta);\
  VST1Q_I32(c_tmp, cq1); VST1Q_I32(c_tmp + 4, cq2);\
  VST1Q_I32(c_tmp + ldc, cq3); VST1Q_I32(c_tmp + ldc + 4, cq4);\
  c_tmp += ldc * 2;

#define SAVE_M8N2 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M8N2(cq1, cq2, cq3, cq4)

#define SAVE_M2N8 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M2N4(cq1, cq3) UNIT_SAVE_M2N4(cq2, cq4)

#define KERNEL_M8N4_UNIT(a_head, b_head) \
  I32X4 cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = cq7 = cq8 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, aq2, bq1;\
  if (kdiv4_left > 0) {\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4); a_rd += 8;\
    bq1 = VLD1Q(b_rd); b_rd += 4;\
  }\
  for (; kdiv4_left > 1; kdiv4_left--) {\
    cq1 = VDOTQ_LANEQ_I32(cq1, aq1, bq1, 0);\
    cq3 = VDOTQ_LANEQ_I32(cq3, aq1, bq1, 1);\
    cq5 = VDOTQ_LANEQ_I32(cq5, aq1, bq1, 2);\
    cq7 = VDOTQ_LANEQ_I32(cq7, aq1, bq1, 3);\
    aq1 = VLD1Q(a_rd);\
    cq2 = VDOTQ_LANEQ_I32(cq2, aq2, bq1, 0);\
    cq4 = VDOTQ_LANEQ_I32(cq4, aq2, bq1, 1);\
    cq6 = VDOTQ_LANEQ_I32(cq6, aq2, bq1, 2);\
    cq8 = VDOTQ_LANEQ_I32(cq8, aq2, bq1, 3);\
    aq2 = VLD1Q(a_rd + 4); a_rd += 8;\
    bq1 = VLD1Q(b_rd); b_rd += 4;\
  }\
  if (kdiv4_left > 0) {\
    cq1 = VDOTQ_LANEQ_I32(cq1, aq1, bq1, 0);\
    cq3 = VDOTQ_LANEQ_I32(cq3, aq1, bq1, 1);\
    cq5 = VDOTQ_LANEQ_I32(cq5, aq1, bq1, 2);\
    cq7 = VDOTQ_LANEQ_I32(cq7, aq1, bq1, 3);\
    cq2 = VDOTQ_LANEQ_I32(cq2, aq2, bq1, 0);\
    cq4 = VDOTQ_LANEQ_I32(cq4, aq2, bq1, 1);\
    cq6 = VDOTQ_LANEQ_I32(cq6, aq2, bq1, 2);\
    cq8 = VDOTQ_LANEQ_I32(cq8, aq2, bq1, 3);\
  }

#define KERNEL_M8N4 KERNEL_M8N4_UNIT(a_head, b_head)
#define KERNEL_M4N8 KERNEL_M8N4_UNIT(b_head, a_head)

#define SAVE_M8N4 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M8N2(cq1, cq2, cq3, cq4) UNIT_SAVE_M8N2(cq5, cq6, cq7, cq8)

#define UNIT_SAVE_M4N4_TRANS(cq1, cq2, cq3, cq4) {\
  I32X4 l1 = VLD1Q_I32(c_tmp);\
  I32X4 l2 = VLD1Q_I32(c_tmp + ldc);\
  I32X4 l3 = VLD1Q_I32(c_tmp + ldc * 2);\
  I32X4 l4 = VLD1Q_I32(c_tmp + ldc * 3);\
  I64X2 t1 = VREINTERPRETQ_I64_I32(VZIP1Q_I32(cq1, cq2));\
  I64X2 t2 = VREINTERPRETQ_I64_I32(VZIP1Q_I32(cq3, cq4));\
  I64X2 t3 = VREINTERPRETQ_I64_I32(VZIP2Q_I32(cq1, cq2));\
  I64X2 t4 = VREINTERPRETQ_I64_I32(VZIP2Q_I32(cq3, cq4));\
  cq1 = VREINTERPRETQ_I32_I64(VZIP1Q_I64(t1, t2));\
  cq2 = VREINTERPRETQ_I32_I64(VZIP2Q_I64(t1, t2));\
  cq3 = VREINTERPRETQ_I32_I64(VZIP1Q_I64(t3, t4));\
  cq4 = VREINTERPRETQ_I32_I64(VZIP2Q_I64(t3, t4));\
  cq1 = VMLAQ_N_I32(cq1, l1, beta); cq2 = VMLAQ_N_I32(cq2, l2, beta);\
  cq3 = VMLAQ_N_I32(cq3, l3, beta); cq4 = VMLAQ_N_I32(cq4, l4, beta);\
  VST1Q_I32(c_tmp, cq1); VST1Q_I32(c_tmp + ldc, cq2);\
  VST1Q_I32(c_tmp + ldc * 2, cq3); VST1Q_I32(c_tmp + ldc * 3, cq4);\
  c_tmp += ldc * 4;\
}
  
#define SAVE_M4N8 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M4N4_TRANS(cq1, cq3, cq5, cq7)\
  UNIT_SAVE_M4N4_TRANS(cq2, cq4, cq6, cq8)

#define KERNEL_M8N8 \
  I32X4 cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  I32X4 cq09, cq10, cq11, cq12, cq13, cq14, cq15, cq16;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = cq07 = cq08 = VDUPQ_N_I32(0);\
  cq09 = cq10 = cq11 = cq12 = cq13 = cq14 = cq15 = cq16 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, aq2, bq1, bq2;\
  if (kdiv4_left > 0) {\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4); a_rd += 8;\
    bq1 = VLD1Q(b_rd); bq2 = VLD1Q(b_rd + 4); b_rd += 8;\
  }\
  for (; kdiv4_left > 1; kdiv4_left--) {\
    cq01 = VDOTQ_LANEQ_I32(cq01, aq1, bq1, 0);\
    cq03 = VDOTQ_LANEQ_I32(cq03, aq1, bq1, 1);\
    cq05 = VDOTQ_LANEQ_I32(cq05, aq1, bq1, 2);\
    cq07 = VDOTQ_LANEQ_I32(cq07, aq1, bq1, 3);\
    cq09 = VDOTQ_LANEQ_I32(cq09, aq1, bq2, 0);\
    cq11 = VDOTQ_LANEQ_I32(cq11, aq1, bq2, 1);\
    cq13 = VDOTQ_LANEQ_I32(cq13, aq1, bq2, 2);\
    cq15 = VDOTQ_LANEQ_I32(cq15, aq1, bq2, 3);\
    aq1 = VLD1Q(a_rd);\
    cq02 = VDOTQ_LANEQ_I32(cq02, aq2, bq1, 0);\
    cq04 = VDOTQ_LANEQ_I32(cq04, aq2, bq1, 1);\
    cq06 = VDOTQ_LANEQ_I32(cq06, aq2, bq1, 2);\
    cq08 = VDOTQ_LANEQ_I32(cq08, aq2, bq1, 3);\
    bq1 = VLD1Q(b_rd);\
    cq10 = VDOTQ_LANEQ_I32(cq10, aq2, bq2, 0);\
    cq12 = VDOTQ_LANEQ_I32(cq12, aq2, bq2, 1);\
    cq14 = VDOTQ_LANEQ_I32(cq14, aq2, bq2, 2);\
    cq16 = VDOTQ_LANEQ_I32(cq16, aq2, bq2, 3);\
    aq2 = VLD1Q(a_rd + 4); a_rd += 8;\
    bq2 = VLD1Q(b_rd + 4); b_rd += 8;\
  }\
  if (kdiv4_left > 0) {\
    cq01 = VDOTQ_LANEQ_I32(cq01, aq1, bq1, 0);\
    cq03 = VDOTQ_LANEQ_I32(cq03, aq1, bq1, 1);\
    cq05 = VDOTQ_LANEQ_I32(cq05, aq1, bq1, 2);\
    cq07 = VDOTQ_LANEQ_I32(cq07, aq1, bq1, 3);\
    cq09 = VDOTQ_LANEQ_I32(cq09, aq1, bq2, 0);\
    cq11 = VDOTQ_LANEQ_I32(cq11, aq1, bq2, 1);\
    cq13 = VDOTQ_LANEQ_I32(cq13, aq1, bq2, 2);\
    cq15 = VDOTQ_LANEQ_I32(cq15, aq1, bq2, 3);\
    cq02 = VDOTQ_LANEQ_I32(cq02, aq2, bq1, 0);\
    cq04 = VDOTQ_LANEQ_I32(cq04, aq2, bq1, 1);\
    cq06 = VDOTQ_LANEQ_I32(cq06, aq2, bq1, 2);\
    cq08 = VDOTQ_LANEQ_I32(cq08, aq2, bq1, 3);\
    cq10 = VDOTQ_LANEQ_I32(cq10, aq2, bq2, 0);\
    cq12 = VDOTQ_LANEQ_I32(cq12, aq2, bq2, 1);\
    cq14 = VDOTQ_LANEQ_I32(cq14, aq2, bq2, 2);\
    cq16 = VDOTQ_LANEQ_I32(cq16, aq2, bq2, 3);\
  }

#define SAVE_M8N8 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M8N2(cq01, cq02, cq03, cq04)\
  UNIT_SAVE_M8N2(cq05, cq06, cq07, cq08)\
  UNIT_SAVE_M8N2(cq09, cq10, cq11, cq12)\
  UNIT_SAVE_M8N2(cq13, cq14, cq15, cq16)

#define KERNEL_M12N1_UNIT(a_head, b_head) \
  I32X4 cq1, cq2, cq3, cq4, cq5, cq6;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, aq2, aq3, aq4, aq5, aq6;\
  I8X8 bd1;\
  if (kdiv4_left > 1) {\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4);\
    aq3 = VLD1Q(a_rd + 8); aq4 = VLD1Q(a_rd + 12);\
    aq5 = VLD1Q(a_rd + 16); aq6 = VLD1Q(a_rd + 20); a_rd += 24;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  for (; kdiv4_left > 3; kdiv4_left -= 2) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0); aq1 = VLD1Q(a_rd);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0); aq2 = VLD1Q(a_rd + 4);\
    cq3 = VDOTQ_LANE_I32(cq3, aq3, bd1, 0); aq3 = VLD1Q(a_rd + 8);\
    cq4 = VDOTQ_LANE_I32(cq4, aq4, bd1, 1); aq4 = VLD1Q(a_rd + 12);\
    cq5 = VDOTQ_LANE_I32(cq5, aq5, bd1, 1); aq5 = VLD1Q(a_rd + 16);\
    cq6 = VDOTQ_LANE_I32(cq6, aq6, bd1, 1); aq6 = VLD1Q(a_rd + 20);\
    a_rd += 24; bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  if (kdiv4_left > 1) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0);\
    cq3 = VDOTQ_LANE_I32(cq3, aq3, bd1, 0);\
    cq4 = VDOTQ_LANE_I32(cq4, aq4, bd1, 1);\
    cq5 = VDOTQ_LANE_I32(cq5, aq5, bd1, 1);\
    cq6 = VDOTQ_LANE_I32(cq6, aq6, bd1, 1);\
    kdiv4_left -= 2;\
  }\
  cq1 = VADDQ_I32(cq1, cq4);\
  cq2 = VADDQ_I32(cq2, cq5);\
  cq3 = VADDQ_I32(cq3, cq6);\
  if (kdiv4_left > 0) {\
    UNALIGNED_LD4B_SREG(bd1, b_rd); b_rd++;\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4);\
    aq3 = VLD1Q(a_rd + 8); a_rd += 12;\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0);\
    cq3 = VDOTQ_LANE_I32(cq3, aq3, bd1, 0);\
  }

#define KERNEL_M12N1 KERNEL_M12N1_UNIT(a_head, b_head)
#define KERNEL_M1N12 KERNEL_M12N1_UNIT(b_head, a_head)

#define SAVE_M12N1 \
  cq1 = VMLAQ_N_I32(cq1, VLD1Q_I32(c_ptr), beta);\
  cq2 = VMLAQ_N_I32(cq2, VLD1Q_I32(c_ptr + 4), beta);\
  cq3 = VMLAQ_N_I32(cq3, VLD1Q_I32(c_ptr + 8), beta);\
  VST1Q_I32(c_ptr, cq1); VST1Q_I32(c_ptr + 4, cq2); VST1Q_I32(c_ptr + 8, cq3);

#define SAVE_M1N12 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M1N4(cq1)\
  UNIT_SAVE_M1N4(cq2) UNIT_SAVE_M1N4(cq3)

#define KERNEL_M12N2_UNIT(a_head, b_head) \
  I32X4 cq1, cq2, cq3, cq4, cq5, cq6;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, aq2, aq3;\
  I8X8 bd1;\
  if (kdiv4_left > 0) {\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4);\
    aq3 = VLD1Q(a_rd + 8); a_rd += 12;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  for (; kdiv4_left > 1; kdiv4_left--) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq4 = VDOTQ_LANE_I32(cq4, aq1, bd1, 1);\
    aq1 = VLD1Q(a_rd);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0);\
    cq5 = VDOTQ_LANE_I32(cq5, aq2, bd1, 1);\
    aq2 = VLD1Q(a_rd + 4);\
    cq3 = VDOTQ_LANE_I32(cq3, aq3, bd1, 0);\
    cq6 = VDOTQ_LANE_I32(cq6, aq3, bd1, 1);\
    aq3 = VLD1Q(a_rd + 8); a_rd += 12;\
    bd1 = VLD1(b_rd); b_rd += 2;\
  }\
  if (kdiv4_left > 0) {\
    cq1 = VDOTQ_LANE_I32(cq1, aq1, bd1, 0);\
    cq4 = VDOTQ_LANE_I32(cq4, aq1, bd1, 1);\
    cq2 = VDOTQ_LANE_I32(cq2, aq2, bd1, 0);\
    cq5 = VDOTQ_LANE_I32(cq5, aq2, bd1, 1);\
    cq3 = VDOTQ_LANE_I32(cq3, aq3, bd1, 0);\
    cq6 = VDOTQ_LANE_I32(cq6, aq3, bd1, 1);\
  }

#define KERNEL_M12N2 KERNEL_M12N2_UNIT(a_head, b_head)
#define KERNEL_M2N12 KERNEL_M12N2_UNIT(b_head, a_head)

#define UNIT_SAVE_M12N2(cq1, cq2, cq3, cq4, cq5, cq6) \
  cq1 = VMLAQ_N_I32(cq1, VLD1Q_I32(c_tmp), beta);\
  cq2 = VMLAQ_N_I32(cq2, VLD1Q_I32(c_tmp + 4), beta);\
  cq3 = VMLAQ_N_I32(cq3, VLD1Q_I32(c_tmp + 8), beta);\
  cq4 = VMLAQ_N_I32(cq4, VLD1Q_I32(c_tmp + ldc), beta);\
  cq5 = VMLAQ_N_I32(cq5, VLD1Q_I32(c_tmp + ldc + 4), beta);\
  cq6 = VMLAQ_N_I32(cq6, VLD1Q_I32(c_tmp + ldc + 8), beta);\
  VST1Q_I32(c_tmp, cq1); VST1Q_I32(c_tmp + 4, cq2);\
  VST1Q_I32(c_tmp + 8, cq3); VST1Q_I32(c_tmp + ldc, cq4);\
  VST1Q_I32(c_tmp + ldc + 4, cq5); VST1Q_I32(c_tmp + ldc + 8, cq6);\
  c_tmp += ldc * 2;

#define SAVE_M12N2 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M12N2(cq1, cq2, cq3, cq4, cq5, cq6)

#define SAVE_M2N12 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M2N4(cq1, cq4) UNIT_SAVE_M2N4(cq2, cq5)\
  UNIT_SAVE_M2N4(cq3, cq6)

#define KERNEL_M12N4_UNIT(a_head, b_head) \
  I32X4 cq01, cq02, cq03, cq04, cq05, cq06;\
  I32X4 cq07, cq08, cq09, cq10, cq11, cq12;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = VDUPQ_N_I32(0);\
  cq07 = cq08 = cq09 = cq10 = cq11 = cq12 = VDUPQ_N_I32(0);\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  I8X16 aq1, aq2, aq3, bq1;\
  if (kdiv4_left > 0) {\
    aq1 = VLD1Q(a_rd); aq2 = VLD1Q(a_rd + 4);\
    aq3 = VLD1Q(a_rd + 8); a_rd += 12;\
    bq1 = VLD1Q(b_rd); b_rd += 4;\
  }\
  for (; kdiv4_left > 1; kdiv4_left--) {\
    cq01 = VDOTQ_LANEQ_I32(cq01, aq1, bq1, 0);\
    cq04 = VDOTQ_LANEQ_I32(cq04, aq1, bq1, 1);\
    cq07 = VDOTQ_LANEQ_I32(cq07, aq1, bq1, 2);\
    cq10 = VDOTQ_LANEQ_I32(cq10, aq1, bq1, 3);\
    aq1 = VLD1Q(a_rd);\
    cq02 = VDOTQ_LANEQ_I32(cq02, aq2, bq1, 0);\
    cq05 = VDOTQ_LANEQ_I32(cq05, aq2, bq1, 1);\
    cq08 = VDOTQ_LANEQ_I32(cq08, aq2, bq1, 2);\
    cq11 = VDOTQ_LANEQ_I32(cq11, aq2, bq1, 3);\
    aq2 = VLD1Q(a_rd + 4);\
    cq03 = VDOTQ_LANEQ_I32(cq03, aq3, bq1, 0);\
    cq06 = VDOTQ_LANEQ_I32(cq06, aq3, bq1, 1);\
    cq09 = VDOTQ_LANEQ_I32(cq09, aq3, bq1, 2);\
    cq12 = VDOTQ_LANEQ_I32(cq12, aq3, bq1, 3);\
    aq3 = VLD1Q(a_rd + 8); a_rd += 12;\
    bq1 = VLD1Q(b_rd); b_rd += 4;\
  }\
  if (kdiv4_left > 0) {\
    cq01 = VDOTQ_LANEQ_I32(cq01, aq1, bq1, 0);\
    cq04 = VDOTQ_LANEQ_I32(cq04, aq1, bq1, 1);\
    cq07 = VDOTQ_LANEQ_I32(cq07, aq1, bq1, 2);\
    cq10 = VDOTQ_LANEQ_I32(cq10, aq1, bq1, 3);\
    cq02 = VDOTQ_LANEQ_I32(cq02, aq2, bq1, 0);\
    cq05 = VDOTQ_LANEQ_I32(cq05, aq2, bq1, 1);\
    cq08 = VDOTQ_LANEQ_I32(cq08, aq2, bq1, 2);\
    cq11 = VDOTQ_LANEQ_I32(cq11, aq2, bq1, 3);\
    cq03 = VDOTQ_LANEQ_I32(cq03, aq3, bq1, 0);\
    cq06 = VDOTQ_LANEQ_I32(cq06, aq3, bq1, 1);\
    cq09 = VDOTQ_LANEQ_I32(cq09, aq3, bq1, 2);\
    cq12 = VDOTQ_LANEQ_I32(cq12, aq3, bq1, 3);\
  }

#define KERNEL_M12N4 KERNEL_M12N4_UNIT(a_head, b_head)
#define KERNEL_M4N12 KERNEL_M12N4_UNIT(b_head, a_head)

#define SAVE_M12N4 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M12N2(cq01, cq02, cq03, cq04, cq05, cq06)\
  UNIT_SAVE_M12N2(cq07, cq08, cq09, cq10, cq11, cq12)

#define SAVE_M4N12 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M4N4_TRANS(cq01, cq04, cq07, cq10)\
  UNIT_SAVE_M4N4_TRANS(cq02, cq05, cq08, cq11)\
  UNIT_SAVE_M4N4_TRANS(cq03, cq06, cq09, cq12)

#define LDQ_STEP1_IDX_A55(v, ptr, idx) "ldr d"#v",["#ptr"],#"#idx"\n\t"
#define LDQ_STEP1_OFF_A55(v, ptr, off) "ldr d"#v",["#ptr",#"#off"]\n\t"
#define LDQ_STEP2_A55(r, ptr, off) "ldr x"#r",["#ptr",#"#off"]\n\t"
#define LDQ_STEP3_A55(r, v) "fmov v"#v".d[1],x"#r"\n\t"
#define LDQ_STEP1_IDX_A76(v, ptr, idx) "ldr q"#v",["#ptr"],#"#idx"\n\t"
#define LDQ_STEP1_OFF_A76(v, ptr, off) "ldr q"#v",["#ptr",#"#off"]\n\t"
#define LDQ_STEP2_A76(r, ptr, off) ""
#define LDQ_STEP3_A76(r, v) ""

#define KERNEL_M8N12_TEMPLATE(cpu) \
  I32 *c_pref = c_ptr + 7; PREF_N12\
  I32X4 cq01, cq02, cq03, cq04, cq05, cq06;\
  I32X4 cq07, cq08, cq09, cq10, cq11, cq12;\
  I32X4 cq13, cq14, cq15, cq16, cq17, cq18;\
  I32X4 cq19, cq20, cq21, cq22, cq23, cq24;\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  __asm__ __volatile__(\
    "movi %[cq01].16b,#0; movi %[cq02].16b,#0\n\t"\
    "movi %[cq03].16b,#0; movi %[cq04].16b,#0\n\t"\
    "movi %[cq05].16b,#0; movi %[cq06].16b,#0\n\t"\
    "movi %[cq07].16b,#0; movi %[cq08].16b,#0\n\t"\
    "movi %[cq09].16b,#0; movi %[cq10].16b,#0\n\t"\
    "movi %[cq11].16b,#0; movi %[cq12].16b,#0\n\t"\
    "movi %[cq13].16b,#0; movi %[cq14].16b,#0\n\t"\
    "movi %[cq15].16b,#0; movi %[cq16].16b,#0\n\t"\
    "movi %[cq17].16b,#0; movi %[cq18].16b,#0\n\t"\
    "movi %[cq19].16b,#0; movi %[cq20].16b,#0\n\t"\
    "movi %[cq21].16b,#0; movi %[cq22].16b,#0\n\t"\
    "movi %[cq23].16b,#0; movi %[cq24].16b,#0\n\t"\
    "cmp %w[kdiv4_left],#1; b.lt 4f\n\t"\
    "ldr q0,[%[a_rd]]; ldr q1,[%[a_rd],#16]; add %[a_rd],%[a_rd],#32\n\t"\
    "ldr q4,[%[b_rd]]; ldr q5,[%[b_rd],#16]; add %[b_rd],%[b_rd],#48\n\t"\
    "cmp %w[kdiv4_left],#3; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    ""IDOT" %[cq01].4s,v0.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(6, %[b_rd], -16)\
    ""IDOT" %[cq02].4s,v1.16b,v4.4b[0]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -8)\
    ""IDOT" %[cq03].4s,v0.16b,v4.4b[1]\n\t" LDQ_STEP1_IDX_##cpu(2, %[a_rd], 64)\
    ""IDOT" %[cq04].4s,v1.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq05].4s,v0.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq06].4s,v1.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq07].4s,v0.16b,v4.4b[3]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -56)\
    ""IDOT" %[cq08].4s,v1.16b,v4.4b[3]\n\t" LDQ_STEP3_##cpu(1, 6)\
    ""IDOT" %[cq09].4s,v0.16b,v5.4b[0]\n\t" LDQ_STEP1_IDX_##cpu(4, %[b_rd], 96)\
    ""IDOT" %[cq10].4s,v1.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq11].4s,v0.16b,v5.4b[1]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -88)\
    ""IDOT" %[cq12].4s,v1.16b,v5.4b[1]\n\t" LDQ_STEP3_##cpu(0, 2)\
    ""IDOT" %[cq13].4s,v0.16b,v5.4b[2]\n\t" LDQ_STEP1_OFF_##cpu(3, %[a_rd], -48)\
    ""IDOT" %[cq14].4s,v1.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq15].4s,v0.16b,v5.4b[3]\n\t"\
    ""IDOT" %[cq16].4s,v1.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(1, 4)\
    ""IDOT" %[cq17].4s,v0.16b,v6.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(5, %[b_rd], -80)\
    ""IDOT" %[cq18].4s,v1.16b,v6.4b[0]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -72)\
    ""IDOT" %[cq19].4s,v0.16b,v6.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -40)\
    ""IDOT" %[cq20].4s,v1.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq21].4s,v0.16b,v6.4b[2]\n\t"\
    ""IDOT" %[cq22].4s,v1.16b,v6.4b[2]\n\t" LDQ_STEP3_##cpu(0, 3)\
    ""IDOT" %[cq23].4s,v0.16b,v6.4b[3]\n\t"\
    ""IDOT" %[cq24].4s,v1.16b,v6.4b[3]\n\t" LDQ_STEP3_##cpu(1, 5)\
    ""IDOT" %[cq01].4s,v2.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(6, %[b_rd], -64)\
    ""IDOT" %[cq02].4s,v3.16b,v4.4b[0]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -56)\
    ""IDOT" %[cq03].4s,v2.16b,v4.4b[1]\n\t" LDQ_STEP1_OFF_##cpu(0, %[a_rd], -32)\
    ""IDOT" %[cq04].4s,v3.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq05].4s,v2.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq06].4s,v3.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq07].4s,v2.16b,v4.4b[3]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -24)\
    ""IDOT" %[cq08].4s,v3.16b,v4.4b[3]\n\t" LDQ_STEP3_##cpu(1, 6)\
    ""IDOT" %[cq09].4s,v2.16b,v5.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(4, %[b_rd], -48)\
    ""IDOT" %[cq10].4s,v3.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq11].4s,v2.16b,v5.4b[1]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -40)\
    ""IDOT" %[cq12].4s,v3.16b,v5.4b[1]\n\t" LDQ_STEP3_##cpu(0, 0)\
    ""IDOT" %[cq13].4s,v2.16b,v5.4b[2]\n\t" LDQ_STEP1_OFF_##cpu(1, %[a_rd], -16)\
    ""IDOT" %[cq14].4s,v3.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq15].4s,v2.16b,v5.4b[3]\n\t"\
    ""IDOT" %[cq16].4s,v3.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(1, 4)\
    ""IDOT" %[cq17].4s,v2.16b,v6.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(5, %[b_rd], -32)\
    ""IDOT" %[cq18].4s,v3.16b,v6.4b[0]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -24)\
    ""IDOT" %[cq19].4s,v2.16b,v6.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -8)\
    ""IDOT" %[cq20].4s,v3.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq21].4s,v2.16b,v6.4b[2]; sub %w[kdiv4_left],%w[kdiv4_left],#2\n\t"\
    ""IDOT" %[cq22].4s,v3.16b,v6.4b[2]\n\t" LDQ_STEP3_##cpu(0, 1)\
    ""IDOT" %[cq23].4s,v2.16b,v6.4b[3]; cmp %w[kdiv4_left],#3\n\t"\
    ""IDOT" %[cq24].4s,v3.16b,v6.4b[3]\n\t" LDQ_STEP3_##cpu(1, 5)\
    "b.ge 1b; 2:\n\t"\
    "cmp %w[kdiv4_left],#2; b.lt 3f\n\t"\
    ""IDOT" %[cq01].4s,v0.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(6, %[b_rd], -16)\
    ""IDOT" %[cq02].4s,v1.16b,v4.4b[0]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -8)\
    ""IDOT" %[cq03].4s,v0.16b,v4.4b[1]\n\t" LDQ_STEP1_IDX_##cpu(2, %[a_rd], 32)\
    ""IDOT" %[cq04].4s,v1.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq05].4s,v0.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq06].4s,v1.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq07].4s,v0.16b,v4.4b[3]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -24)\
    ""IDOT" %[cq08].4s,v1.16b,v4.4b[3]\n\t" LDQ_STEP3_##cpu(1, 6)\
    ""IDOT" %[cq09].4s,v0.16b,v5.4b[0]\n\t" LDQ_STEP1_IDX_##cpu(4, %[b_rd], 48)\
    ""IDOT" %[cq10].4s,v1.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq11].4s,v0.16b,v5.4b[1]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -40)\
    ""IDOT" %[cq12].4s,v1.16b,v5.4b[1]\n\t" LDQ_STEP3_##cpu(0, 2)\
    ""IDOT" %[cq13].4s,v0.16b,v5.4b[2]\n\t" LDQ_STEP1_OFF_##cpu(3, %[a_rd], -16)\
    ""IDOT" %[cq14].4s,v1.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq15].4s,v0.16b,v5.4b[3]\n\t"\
    ""IDOT" %[cq16].4s,v1.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(1, 4)\
    ""IDOT" %[cq17].4s,v0.16b,v6.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(5, %[b_rd], -32)\
    ""IDOT" %[cq18].4s,v1.16b,v6.4b[0]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -24)\
    ""IDOT" %[cq19].4s,v0.16b,v6.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -8)\
    ""IDOT" %[cq20].4s,v1.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq21].4s,v0.16b,v6.4b[2]\n\t"\
    ""IDOT" %[cq22].4s,v1.16b,v6.4b[2]\n\t" LDQ_STEP3_##cpu(0, 3)\
    ""IDOT" %[cq23].4s,v0.16b,v6.4b[3]\n\t"\
    ""IDOT" %[cq24].4s,v1.16b,v6.4b[3]\n\t" LDQ_STEP3_##cpu(1, 5)\
    ""IDOT" %[cq01].4s,v2.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(6, %[b_rd], -16)\
    ""IDOT" %[cq02].4s,v3.16b,v4.4b[0]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -8)\
    ""IDOT" %[cq03].4s,v2.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq04].4s,v3.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq05].4s,v2.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq06].4s,v3.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq07].4s,v2.16b,v4.4b[3]\n\t"\
    ""IDOT" %[cq08].4s,v3.16b,v4.4b[3]\n\t" LDQ_STEP3_##cpu(1, 6)\
    ""IDOT" %[cq09].4s,v2.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq10].4s,v3.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq11].4s,v2.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq12].4s,v3.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq13].4s,v2.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq14].4s,v3.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq15].4s,v2.16b,v5.4b[3]\n\t"\
    ""IDOT" %[cq16].4s,v3.16b,v5.4b[3]\n\t"\
    ""IDOT" %[cq17].4s,v2.16b,v6.4b[0]\n\t"\
    ""IDOT" %[cq18].4s,v3.16b,v6.4b[0]\n\t"\
    ""IDOT" %[cq19].4s,v2.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq20].4s,v3.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq21].4s,v2.16b,v6.4b[2]\n\t"\
    ""IDOT" %[cq22].4s,v3.16b,v6.4b[2]\n\t"\
    ""IDOT" %[cq23].4s,v2.16b,v6.4b[3]\n\t"\
    ""IDOT" %[cq24].4s,v3.16b,v6.4b[3]\n\t"\
    "b 4f; 3:\n\t"\
    ""IDOT" %[cq01].4s,v0.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(6, %[b_rd], -16)\
    ""IDOT" %[cq02].4s,v1.16b,v4.4b[0]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -8)\
    ""IDOT" %[cq03].4s,v0.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq04].4s,v1.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq05].4s,v0.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq06].4s,v1.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq07].4s,v0.16b,v4.4b[3]\n\t"\
    ""IDOT" %[cq08].4s,v1.16b,v4.4b[3]\n\t" LDQ_STEP3_##cpu(1, 6)\
    ""IDOT" %[cq09].4s,v0.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq10].4s,v1.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq11].4s,v0.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq12].4s,v1.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq13].4s,v0.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq14].4s,v1.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq15].4s,v0.16b,v5.4b[3]\n\t"\
    ""IDOT" %[cq16].4s,v1.16b,v5.4b[3]\n\t"\
    ""IDOT" %[cq17].4s,v0.16b,v6.4b[0]\n\t"\
    ""IDOT" %[cq18].4s,v1.16b,v6.4b[0]\n\t"\
    ""IDOT" %[cq19].4s,v0.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq20].4s,v1.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq21].4s,v0.16b,v6.4b[2]\n\t"\
    ""IDOT" %[cq22].4s,v1.16b,v6.4b[2]\n\t"\
    ""IDOT" %[cq23].4s,v0.16b,v6.4b[3]\n\t"\
    ""IDOT" %[cq24].4s,v1.16b,v6.4b[3]\n\t"\
    "4:\n\t"\
  :[cq01]"=w"(cq01),[cq02]"=w"(cq02),[cq03]"=w"(cq03),[cq04]"=w"(cq04),\
  [cq05]"=w"(cq05),[cq06]"=w"(cq06),[cq07]"=w"(cq07),[cq08]"=w"(cq08),\
  [cq09]"=w"(cq09),[cq10]"=w"(cq10),[cq11]"=w"(cq11),[cq12]"=w"(cq12),\
  [cq13]"=w"(cq13),[cq14]"=w"(cq14),[cq15]"=w"(cq15),[cq16]"=w"(cq16),\
  [cq17]"=w"(cq17),[cq18]"=w"(cq18),[cq19]"=w"(cq19),[cq20]"=w"(cq20),\
  [cq21]"=w"(cq21),[cq22]"=w"(cq22),[cq23]"=w"(cq23),[cq24]"=w"(cq24),\
  [kdiv4_left]"+r"(kdiv4_left),[a_rd]"+r"(a_rd),[b_rd]"+r"(b_rd)\
  ::"cc","memory","x0","x1","v0","v1","v2","v3","v4","v5","v6");

#define SAVE_M8N12 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M8N2(cq01, cq02, cq03, cq04)\
  UNIT_SAVE_M8N2(cq05, cq06, cq07, cq08)\
  UNIT_SAVE_M8N2(cq09, cq10, cq11, cq12)\
  UNIT_SAVE_M8N2(cq13, cq14, cq15, cq16)\
  UNIT_SAVE_M8N2(cq17, cq18, cq19, cq20)\
  UNIT_SAVE_M8N2(cq21, cq22, cq23, cq24)

#define KERNEL_M12N8_TEMPLATE(cpu) \
  I32 *c_pref = c_ptr + 11; PREF_N8\
  I32X4 cq01, cq02, cq03, cq04, cq05, cq06;\
  I32X4 cq07, cq08, cq09, cq10, cq11, cq12;\
  I32X4 cq13, cq14, cq15, cq16, cq17, cq18;\
  I32X4 cq19, cq20, cq21, cq22, cq23, cq24;\
  NORMAL_KERNEL_SETUP(a_head, b_head)\
  __asm__ __volatile__(\
    "movi %[cq01].16b,#0; movi %[cq02].16b,#0\n\t"\
    "movi %[cq03].16b,#0; movi %[cq04].16b,#0\n\t"\
    "movi %[cq05].16b,#0; movi %[cq06].16b,#0\n\t"\
    "movi %[cq07].16b,#0; movi %[cq08].16b,#0\n\t"\
    "movi %[cq09].16b,#0; movi %[cq10].16b,#0\n\t"\
    "movi %[cq11].16b,#0; movi %[cq12].16b,#0\n\t"\
    "movi %[cq13].16b,#0; movi %[cq14].16b,#0\n\t"\
    "movi %[cq15].16b,#0; movi %[cq16].16b,#0\n\t"\
    "movi %[cq17].16b,#0; movi %[cq18].16b,#0\n\t"\
    "movi %[cq19].16b,#0; movi %[cq20].16b,#0\n\t"\
    "movi %[cq21].16b,#0; movi %[cq22].16b,#0\n\t"\
    "movi %[cq23].16b,#0; movi %[cq24].16b,#0\n\t"\
    "cmp %w[kdiv4_left],#1; b.lt 4f\n\t"\
    "ldr q0,[%[a_rd]]; ldr q1,[%[a_rd],#16]; add %[a_rd],%[a_rd],#48\n\t"\
    "ldr q4,[%[b_rd]]; ldr q5,[%[b_rd],#16]; add %[b_rd],%[b_rd],#32\n\t"\
    "cmp %w[kdiv4_left],#3; b.lt 2f\n\t"\
    ".balign 16; 1:\n\t"\
    ""IDOT" %[cq01].4s,v0.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(2, %[a_rd], -16)\
    ""IDOT" %[cq04].4s,v0.16b,v4.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -8)\
    ""IDOT" %[cq07].4s,v0.16b,v4.4b[2]\n\t" LDQ_STEP1_IDX_##cpu(6, %[b_rd], 64)\
    ""IDOT" %[cq10].4s,v0.16b,v4.4b[3]\n\t"\
    ""IDOT" %[cq13].4s,v0.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq16].4s,v0.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq19].4s,v0.16b,v5.4b[2]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -56)\
    ""IDOT" %[cq22].4s,v0.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(0, 2)\
    ""IDOT" %[cq02].4s,v1.16b,v4.4b[0]\n\t" LDQ_STEP1_IDX_##cpu(0, %[a_rd], 96)\
    ""IDOT" %[cq05].4s,v1.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq08].4s,v1.16b,v4.4b[2]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -88)\
    ""IDOT" %[cq11].4s,v1.16b,v4.4b[3]\n\t" LDQ_STEP3_##cpu(1, 6)\
    ""IDOT" %[cq14].4s,v1.16b,v5.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(7, %[b_rd], -48)\
    ""IDOT" %[cq17].4s,v1.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq20].4s,v1.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq23].4s,v1.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(0, 0)\
    ""IDOT" %[cq03].4s,v2.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(1, %[a_rd], -80)\
    ""IDOT" %[cq06].4s,v2.16b,v4.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -72)\
    ""IDOT" %[cq09].4s,v2.16b,v4.4b[2]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -40)\
    ""IDOT" %[cq12].4s,v2.16b,v4.4b[3]\n\t"\
    ""IDOT" %[cq15].4s,v2.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq18].4s,v2.16b,v5.4b[1]\n\t" LDQ_STEP3_##cpu(1, 7)\
    ""IDOT" %[cq21].4s,v2.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq24].4s,v2.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(0, 1)\
    ""IDOT" %[cq01].4s,v0.16b,v6.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(2, %[a_rd], -64)\
    ""IDOT" %[cq04].4s,v0.16b,v6.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -56)\
    ""IDOT" %[cq07].4s,v0.16b,v6.4b[2]\n\t" LDQ_STEP1_OFF_##cpu(4, %[b_rd], -32)\
    ""IDOT" %[cq10].4s,v0.16b,v6.4b[3]\n\t"\
    ""IDOT" %[cq13].4s,v0.16b,v7.4b[0]\n\t"\
    ""IDOT" %[cq16].4s,v0.16b,v7.4b[1]\n\t"\
    ""IDOT" %[cq19].4s,v0.16b,v7.4b[2]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -24)\
    ""IDOT" %[cq22].4s,v0.16b,v7.4b[3]\n\t" LDQ_STEP3_##cpu(0, 2)\
    ""IDOT" %[cq02].4s,v1.16b,v6.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(0, %[a_rd], -48)\
    ""IDOT" %[cq05].4s,v1.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq08].4s,v1.16b,v6.4b[2]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -40)\
    ""IDOT" %[cq11].4s,v1.16b,v6.4b[3]\n\t" LDQ_STEP3_##cpu(1, 4)\
    ""IDOT" %[cq14].4s,v1.16b,v7.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(5, %[b_rd], -16)\
    ""IDOT" %[cq17].4s,v1.16b,v7.4b[1]\n\t"\
    ""IDOT" %[cq20].4s,v1.16b,v7.4b[2]\n\t"\
    ""IDOT" %[cq23].4s,v1.16b,v7.4b[3]\n\t" LDQ_STEP3_##cpu(0, 0)\
    ""IDOT" %[cq03].4s,v2.16b,v6.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(1, %[a_rd], -32)\
    ""IDOT" %[cq06].4s,v2.16b,v6.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -24)\
    ""IDOT" %[cq09].4s,v2.16b,v6.4b[2]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -8)\
    ""IDOT" %[cq12].4s,v2.16b,v6.4b[3]\n\t"\
    ""IDOT" %[cq15].4s,v2.16b,v7.4b[0]; sub %w[kdiv4_left],%w[kdiv4_left],#2\n\t"\
    ""IDOT" %[cq18].4s,v2.16b,v7.4b[1]\n\t" LDQ_STEP3_##cpu(1, 5)\
    ""IDOT" %[cq21].4s,v2.16b,v7.4b[2]; cmp %w[kdiv4_left],#3\n\t"\
    ""IDOT" %[cq24].4s,v2.16b,v7.4b[3]\n\t" LDQ_STEP3_##cpu(0, 1)\
    "b.ge 1b; 2:\n\t"\
    "cmp %w[kdiv4_left],#2; b.lt 3f\n\t"\
    ""IDOT" %[cq01].4s,v0.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(2, %[a_rd], -16)\
    ""IDOT" %[cq04].4s,v0.16b,v4.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -8)\
    ""IDOT" %[cq07].4s,v0.16b,v4.4b[2]\n\t" LDQ_STEP1_IDX_##cpu(6, %[b_rd], 32)\
    ""IDOT" %[cq10].4s,v0.16b,v4.4b[3]\n\t"\
    ""IDOT" %[cq13].4s,v0.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq16].4s,v0.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq19].4s,v0.16b,v5.4b[2]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -24)\
    ""IDOT" %[cq22].4s,v0.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(0, 2)\
    ""IDOT" %[cq02].4s,v1.16b,v4.4b[0]\n\t" LDQ_STEP1_IDX_##cpu(0, %[a_rd], 48)\
    ""IDOT" %[cq05].4s,v1.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq08].4s,v1.16b,v4.4b[2]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -40)\
    ""IDOT" %[cq11].4s,v1.16b,v4.4b[3]\n\t" LDQ_STEP3_##cpu(1, 6)\
    ""IDOT" %[cq14].4s,v1.16b,v5.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(7, %[b_rd], -16)\
    ""IDOT" %[cq17].4s,v1.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq20].4s,v1.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq23].4s,v1.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(0, 0)\
    ""IDOT" %[cq03].4s,v2.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(1, %[a_rd], -32)\
    ""IDOT" %[cq06].4s,v2.16b,v4.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -24)\
    ""IDOT" %[cq09].4s,v2.16b,v4.4b[2]\n\t" LDQ_STEP2_##cpu(1, %[b_rd], -8)\
    ""IDOT" %[cq12].4s,v2.16b,v4.4b[3]\n\t"\
    ""IDOT" %[cq15].4s,v2.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq18].4s,v2.16b,v5.4b[1]\n\t" LDQ_STEP3_##cpu(1, 7)\
    ""IDOT" %[cq21].4s,v2.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq24].4s,v2.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(0, 1)\
    ""IDOT" %[cq01].4s,v0.16b,v6.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(2, %[a_rd], -16)\
    ""IDOT" %[cq04].4s,v0.16b,v6.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -8)\
    ""IDOT" %[cq07].4s,v0.16b,v6.4b[2]\n\t"\
    ""IDOT" %[cq10].4s,v0.16b,v6.4b[3]\n\t"\
    ""IDOT" %[cq13].4s,v0.16b,v7.4b[0]\n\t"\
    ""IDOT" %[cq16].4s,v0.16b,v7.4b[1]\n\t"\
    ""IDOT" %[cq19].4s,v0.16b,v7.4b[2]\n\t"\
    ""IDOT" %[cq22].4s,v0.16b,v7.4b[3]\n\t" LDQ_STEP3_##cpu(0, 2)\
    ""IDOT" %[cq02].4s,v1.16b,v6.4b[0]\n\t"\
    ""IDOT" %[cq05].4s,v1.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq08].4s,v1.16b,v6.4b[2]\n\t"\
    ""IDOT" %[cq11].4s,v1.16b,v6.4b[3]\n\t"\
    ""IDOT" %[cq14].4s,v1.16b,v7.4b[0]\n\t"\
    ""IDOT" %[cq17].4s,v1.16b,v7.4b[1]\n\t"\
    ""IDOT" %[cq20].4s,v1.16b,v7.4b[2]\n\t"\
    ""IDOT" %[cq23].4s,v1.16b,v7.4b[3]\n\t"\
    ""IDOT" %[cq03].4s,v2.16b,v6.4b[0]\n\t"\
    ""IDOT" %[cq06].4s,v2.16b,v6.4b[1]\n\t"\
    ""IDOT" %[cq09].4s,v2.16b,v6.4b[2]\n\t"\
    ""IDOT" %[cq12].4s,v2.16b,v6.4b[3]\n\t"\
    ""IDOT" %[cq15].4s,v2.16b,v7.4b[0]\n\t"\
    ""IDOT" %[cq18].4s,v2.16b,v7.4b[1]\n\t"\
    ""IDOT" %[cq21].4s,v2.16b,v7.4b[2]\n\t"\
    ""IDOT" %[cq24].4s,v2.16b,v7.4b[3]\n\t"\
    "b 4f; 3:\n\t"\
    ""IDOT" %[cq01].4s,v0.16b,v4.4b[0]\n\t" LDQ_STEP1_OFF_##cpu(2, %[a_rd], -16)\
    ""IDOT" %[cq04].4s,v0.16b,v4.4b[1]\n\t" LDQ_STEP2_##cpu(0, %[a_rd], -8)\
    ""IDOT" %[cq07].4s,v0.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq10].4s,v0.16b,v4.4b[3]\n\t"\
    ""IDOT" %[cq13].4s,v0.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq16].4s,v0.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq19].4s,v0.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq22].4s,v0.16b,v5.4b[3]\n\t" LDQ_STEP3_##cpu(0, 2)\
    ""IDOT" %[cq02].4s,v1.16b,v4.4b[0]\n\t"\
    ""IDOT" %[cq05].4s,v1.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq08].4s,v1.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq11].4s,v1.16b,v4.4b[3]\n\t"\
    ""IDOT" %[cq14].4s,v1.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq17].4s,v1.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq20].4s,v1.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq23].4s,v1.16b,v5.4b[3]\n\t"\
    ""IDOT" %[cq03].4s,v2.16b,v4.4b[0]\n\t"\
    ""IDOT" %[cq06].4s,v2.16b,v4.4b[1]\n\t"\
    ""IDOT" %[cq09].4s,v2.16b,v4.4b[2]\n\t"\
    ""IDOT" %[cq12].4s,v2.16b,v4.4b[3]\n\t"\
    ""IDOT" %[cq15].4s,v2.16b,v5.4b[0]\n\t"\
    ""IDOT" %[cq18].4s,v2.16b,v5.4b[1]\n\t"\
    ""IDOT" %[cq21].4s,v2.16b,v5.4b[2]\n\t"\
    ""IDOT" %[cq24].4s,v2.16b,v5.4b[3]\n\t"\
    "4:\n\t"\
  :[cq01]"=w"(cq01),[cq02]"=w"(cq02),[cq03]"=w"(cq03),[cq04]"=w"(cq04),\
  [cq05]"=w"(cq05),[cq06]"=w"(cq06),[cq07]"=w"(cq07),[cq08]"=w"(cq08),\
  [cq09]"=w"(cq09),[cq10]"=w"(cq10),[cq11]"=w"(cq11),[cq12]"=w"(cq12),\
  [cq13]"=w"(cq13),[cq14]"=w"(cq14),[cq15]"=w"(cq15),[cq16]"=w"(cq16),\
  [cq17]"=w"(cq17),[cq18]"=w"(cq18),[cq19]"=w"(cq19),[cq20]"=w"(cq20),\
  [cq21]"=w"(cq21),[cq22]"=w"(cq22),[cq23]"=w"(cq23),[cq24]"=w"(cq24),\
  [kdiv4_left]"+r"(kdiv4_left),[a_rd]"+r"(a_rd),[b_rd]"+r"(b_rd)\
  ::"cc","memory","x0","x1","v0","v1","v2","v4","v5","v6","v7");

#define SAVE_M12N8 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M12N2(cq01, cq02, cq03, cq04, cq05, cq06)\
  UNIT_SAVE_M12N2(cq07, cq08, cq09, cq10, cq11, cq12)\
  UNIT_SAVE_M12N2(cq13, cq14, cq15, cq16, cq17, cq18)\
  UNIT_SAVE_M12N2(cq19, cq20, cq21, cq22, cq23, cq24)

#define NEON_IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(mdim, ndim, srcint, dstint) \
static inline void\
  inline_dualpack_gemm_a##srcint##_b##srcint##_c##dstint##_m##mdim##_n##ndim(\
  const srcint *a_head, const srcint *b_head, dstint *c_ptr,\
  uint32_t K, dstint beta, uint32_t ldc) {\
  KERNEL_M##mdim##N##ndim\
  SAVE_M##mdim##N##ndim\
}

#define IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(mdim, ndim, srcint, dstint)\
  NEON_IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(mdim, ndim, srcint, dstint)


IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 1, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 2, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 1, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 2, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 4, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 4, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 1, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 2, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 4, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 8, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 8, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 8, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 1, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 2, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 4, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 8, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 12, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 12, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 12, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 1, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 2, I32, I32)
IDOTGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 4, I32, I32)

#endif
