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
 * File:        NeonI8I32MlaGemmKernel.h
 * Description: Inline kernel function templates for NEON 8->32bit ingeter
 *              GEMM. This template can be used to generate signed and unsigned
 *              integer matmul functions.
 * Names:       "KERNEL_MxNy": code blocks that read panels from souce
 *              matrices and multiply them.
 *              "SAVE_MxNy": code blocks that store the multiply results
 *              to a region of output matrix.
 *****************************************************************************/

#include "arm_neon/NeonIntOpSign.h"

#ifndef INCLUDE_NEON_I8I32_KERNEL
#define INCLUDE_NEON_I8I32_KERNEL

#define COMMON_KERNEL_HEADER(a_head, b_head) \
  const I16 *a_ptr = a_head;\
  const I16 *b_ptr = b_head;\
  uint32_t k_left = K;

#define KERNEL_M1N1 \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, bd1;\
  I32X4 cq1 = VDUPQ_N_I32(0);\
  for (; k_left > 3; k_left -= 4) {\
    ad1 = VLD1_I16(a_ptr); a_ptr += 4;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq1 = VMLAL_I16(cq1, ad1, bd1);\
  }\
  I32X2 cd1 = VADD_I32(VGET_LOW_I32(cq1), VGET_HIGH_I32(cq1));\
  I32 cs1 = VGET_LANE_I32(cd1, 0) + VGET_LANE_I32(cd1, 1);\
  for (; k_left > 0; k_left--) {\
    cs1 += (I32)(*a_ptr++) * (I32)(*b_ptr++);\
  }

#define SAVE_M1N1 \
  cs1 += c_ptr[0] * beta; c_ptr[0] = cs1;

#define KERNEL_M2N1_UNIT(a_head, b_head) \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, bd1;\
  I16X4X2 add1;\
  I32X4 cq1, cq2;\
  cq1 = cq2 = VDUPQ_N_I32(0);\
  for (; k_left > 3; k_left -= 4) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4); a_ptr += 8;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    add1 = VUZP_I16(ad1, ad2);\
    cq1 = VMLAL_I16(cq1, add1.val[0], bd1);\
    cq2 = VMLAL_I16(cq2, add1.val[1], bd1);\
  }\
  I32X2 cd1 = VADD_I32(VGET_LOW_I32(cq1), VGET_HIGH_I32(cq1));\
  I32X2 cd2 = VADD_I32(VGET_LOW_I32(cq2), VGET_HIGH_I32(cq2));\
  I32 cs1 = VGET_LANE_I32(cd1, 0) + VGET_LANE_I32(cd1, 1);\
  I32 cs2 = VGET_LANE_I32(cd2, 0) + VGET_LANE_I32(cd2, 1);\
  for (; k_left > 0; k_left--) {\
    I32 bs1 = *b_ptr++;\
    cs1 += (I32)a_ptr[0] * bs1;\
    cs2 += (I32)a_ptr[1] * bs1;\
    a_ptr += 2;\
  }

#define KERNEL_M2N1 KERNEL_M2N1_UNIT(a_head, b_head)
#define KERNEL_M1N2 KERNEL_M2N1_UNIT(b_head, a_head)

#define SAVE_M2N1 \
  cs1 += c_ptr[0] * beta; cs2 += c_ptr[1] * beta;\
  c_ptr[0] = cs1; c_ptr[1] = cs2;

#define SAVE_M1N2 \
  cs1 += c_ptr[0] * beta; cs2 += c_ptr[ldc] * beta;\
  c_ptr[0] = cs1; c_ptr[ldc] = cs2;

#define KERNEL_M2N2 \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, bd1, bd2;\
  I16X4X2 add1, bdd1;\
  I32X4 cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = VDUPQ_N_I32(0);\
  for (; k_left > 3; k_left -= 4) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4); a_ptr += 8;\
    bd1 = VLD1_I16(b_ptr); bd2 = VLD1_I16(b_ptr + 4); b_ptr += 8;\
    add1 = VUZP_I16(ad1, ad2); bdd1 = VUZP_I16(bd1, bd2);\
    cq1 = VMLAL_I16(cq1, add1.val[0], bdd1.val[0]);\
    cq2 = VMLAL_I16(cq2, add1.val[1], bdd1.val[0]);\
    cq3 = VMLAL_I16(cq3, add1.val[0], bdd1.val[1]);\
    cq4 = VMLAL_I16(cq4, add1.val[1], bdd1.val[1]);\
  }\
  I32X2 cd1 = VADD_I32(VGET_LOW_I32(cq1), VGET_HIGH_I32(cq1));\
  I32X2 cd2 = VADD_I32(VGET_LOW_I32(cq2), VGET_HIGH_I32(cq2));\
  I32X2 cd3 = VADD_I32(VGET_LOW_I32(cq3), VGET_HIGH_I32(cq3));\
  I32X2 cd4 = VADD_I32(VGET_LOW_I32(cq4), VGET_HIGH_I32(cq4));\
  I32 cs1 = VGET_LANE_I32(cd1, 0) + VGET_LANE_I32(cd1, 1);\
  I32 cs2 = VGET_LANE_I32(cd2, 0) + VGET_LANE_I32(cd2, 1);\
  I32 cs3 = VGET_LANE_I32(cd3, 0) + VGET_LANE_I32(cd3, 1);\
  I32 cs4 = VGET_LANE_I32(cd4, 0) + VGET_LANE_I32(cd4, 1);\
  for (; k_left > 0; k_left--) {\
    I32 as1 = a_ptr[0];\
    I32 as2 = a_ptr[1]; a_ptr += 2;\
    I32 bs1 = b_ptr[0];\
    I32 bs2 = b_ptr[1]; b_ptr += 2;\
    cs1 += as1 * bs1; cs2 += as2 * bs1;\
    cs3 += as1 * bs2; cs4 += as2 * bs2;\
  }

#define SAVE_M2N2 \
  I32 *c_l1 = c_ptr + ldc;\
  cs1 += c_ptr[0] * beta; cs2 += c_ptr[1] * beta;\
  cs3 += c_l1[0] * beta; cs4 += c_l1[1] * beta;\
  c_ptr[0] = cs1; c_ptr[1] = cs2;\
  c_l1[0] = cs3; c_l1[1] = cs4;

#define KERNEL_M4N1_UNIT(a_head, b_head) \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, ad3, ad4, bd1;\
  I32X4 cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = VDUPQ_N_I32(0);\
  for (; k_left > 3; k_left -= 4) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); ad4 = VLD1_I16(a_ptr + 12); a_ptr += 16;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq1 = VMLAL_LANE_I16(cq1, ad1, bd1, 0);\
    cq2 = VMLAL_LANE_I16(cq2, ad2, bd1, 1);\
    cq3 = VMLAL_LANE_I16(cq3, ad3, bd1, 2);\
    cq4 = VMLAL_LANE_I16(cq4, ad4, bd1, 3);\
  }\
  cq1 = VADDQ_I32(cq1, cq3); cq2 = VADDQ_I32(cq2, cq4);\
  cq1 = VADDQ_I32(cq1, cq2);\
  for (; k_left > 0; k_left--) {\
    ad1 = VLD1_I16(a_ptr); a_ptr += 4;\
    I16 bs1 = *b_ptr++;\
    cq1 = VMLAL_N_I16(cq1, ad1, bs1);\
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
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, bd1;\
  I32X4 cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = VDUPQ_N_I32(0);\
  for (; k_left > 1; k_left -= 2) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4); a_ptr += 8;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq1 = VMLAL_LANE_I16(cq1, ad1, bd1, 0);\
    cq2 = VMLAL_LANE_I16(cq2, ad1, bd1, 1);\
    cq3 = VMLAL_LANE_I16(cq3, ad2, bd1, 2);\
    cq4 = VMLAL_LANE_I16(cq4, ad2, bd1, 3);\
  }\
  cq1 = VADDQ_I32(cq1, cq3); cq2 = VADDQ_I32(cq2, cq4);\
  for (; k_left > 0; k_left--) {\
    ad1 = VLD1_I16(a_ptr); a_ptr += 4;\
    I16 bs1 = b_ptr[0];\
    I16 bs2 = b_ptr[1]; b_ptr += 2;\
    cq1 = VMLAL_N_I16(cq1, ad1, bs1); cq2 = VMLAL_N_I16(cq2, ad1, bs2);\
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
  I32X4X2 tm1 = VZIPQ_I32(cq1, cq2);\
  I32X2 l1 = VMLA_N_I32(VGET_LOW_I32(tm1.val[0]),\
    VLD1_I32(c_tmp), beta);\
  I32X2 l2 = VMLA_N_I32(VGET_HIGH_I32(tm1.val[0]),\
    VLD1_I32(c_tmp + ldc), beta);\
  VST1_I32(c_tmp, l1); VST1_I32(c_tmp + ldc, l2); c_tmp += ldc * 2;\
  I32X2 l3 = VMLA_N_I32(VGET_LOW_I32(tm1.val[1]),\
    VLD1_I32(c_tmp), beta);\
  I32X2 l4 = VMLA_N_I32(VGET_HIGH_I32(tm1.val[1]),\
    VLD1_I32(c_tmp + ldc), beta);\
  VST1_I32(c_tmp, l3); VST1_I32(c_tmp + ldc, l4); c_tmp += ldc * 2;\
}

#define SAVE_M2N4 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M2N4(cq1, cq2)

#define KERNEL_M4N4 \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, bd1;\
  I32X4 cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = VDUPQ_N_I32(0);\
  for (; k_left > 0; k_left--) {\
    ad1 = VLD1_I16(a_ptr); a_ptr += 4;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq1 = VMLAL_LANE_I16(cq1, ad1, bd1, 0);\
    cq2 = VMLAL_LANE_I16(cq2, ad1, bd1, 1);\
    cq3 = VMLAL_LANE_I16(cq3, ad1, bd1, 2);\
    cq4 = VMLAL_LANE_I16(cq4, ad1, bd1, 3);\
  }

#define SAVE_M4N4 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M4N2(cq1, cq2) UNIT_SAVE_M4N2(cq3, cq4)

#define KERNEL_M8N1_UNIT(a_head, b_head) \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, ad3, ad4, ad5, ad6, ad7, ad8, bd1;\
  I32X4 cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = VDUPQ_N_I32(0);\
  for (; k_left > 3; k_left -= 4) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); ad4 = VLD1_I16(a_ptr + 12);\
    ad5 = VLD1_I16(a_ptr + 16); ad6 = VLD1_I16(a_ptr + 20);\
    ad7 = VLD1_I16(a_ptr + 24); ad8 = VLD1_I16(a_ptr + 28); a_ptr += 32;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq1 = VMLAL_LANE_I16(cq1, ad1, bd1, 0);\
    cq2 = VMLAL_LANE_I16(cq2, ad2, bd1, 0);\
    cq3 = VMLAL_LANE_I16(cq3, ad3, bd1, 1);\
    cq4 = VMLAL_LANE_I16(cq4, ad4, bd1, 1);\
    cq1 = VMLAL_LANE_I16(cq1, ad5, bd1, 2);\
    cq2 = VMLAL_LANE_I16(cq2, ad6, bd1, 2);\
    cq3 = VMLAL_LANE_I16(cq3, ad7, bd1, 3);\
    cq4 = VMLAL_LANE_I16(cq4, ad8, bd1, 3);\
  }\
  cq1 = VADDQ_I32(cq1, cq3); cq2 = VADDQ_I32(cq2, cq4);\
  for (; k_left > 0; k_left--) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4); a_ptr += 8;\
    I16 bs1 = *b_ptr++;\
    cq1 = VMLAL_N_I16(cq1, ad1, bs1); cq2 = VMLAL_N_I16(cq2, ad2, bs1);\
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
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, ad3, ad4, bd1;\
  I32X4 cq1, cq2, cq3, cq4;\
  cq1 = cq2 = cq3 = cq4 = VDUPQ_N_I32(0);\
  for (; k_left > 1; k_left -= 2) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); ad4 = VLD1_I16(a_ptr + 12); a_ptr += 16;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq1 = VMLAL_LANE_I16(cq1, ad1, bd1, 0);\
    cq2 = VMLAL_LANE_I16(cq2, ad2, bd1, 0);\
    cq3 = VMLAL_LANE_I16(cq3, ad1, bd1, 1);\
    cq4 = VMLAL_LANE_I16(cq4, ad2, bd1, 1);\
    cq1 = VMLAL_LANE_I16(cq1, ad3, bd1, 2);\
    cq2 = VMLAL_LANE_I16(cq2, ad4, bd1, 2);\
    cq3 = VMLAL_LANE_I16(cq3, ad3, bd1, 3);\
    cq4 = VMLAL_LANE_I16(cq4, ad4, bd1, 3);\
  }\
  if (k_left > 0) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4); a_ptr += 8;\
    I16 bs1 = b_ptr[0];\
    I16 bs2 = b_ptr[1]; b_ptr += 2;\
    cq1 = VMLAL_N_I16(cq1, ad1, bs1); cq2 = VMLAL_N_I16(cq2, ad2, bs1);\
    cq3 = VMLAL_N_I16(cq3, ad1, bs2); cq4 = VMLAL_N_I16(cq4, ad2, bs2);\
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
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, bd1;\
  I32X4 cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = cq7 = cq8 = VDUPQ_N_I32(0);\
  for (; k_left > 0; k_left--) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4); a_ptr += 8;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq1 = VMLAL_LANE_I16(cq1, ad1, bd1, 0);\
    cq2 = VMLAL_LANE_I16(cq2, ad2, bd1, 0);\
    cq3 = VMLAL_LANE_I16(cq3, ad1, bd1, 1);\
    cq4 = VMLAL_LANE_I16(cq4, ad2, bd1, 1);\
    cq5 = VMLAL_LANE_I16(cq5, ad1, bd1, 2);\
    cq6 = VMLAL_LANE_I16(cq6, ad2, bd1, 2);\
    cq7 = VMLAL_LANE_I16(cq7, ad1, bd1, 3);\
    cq8 = VMLAL_LANE_I16(cq8, ad2, bd1, 3);\
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
  I32X4X2 tm1 = VZIPQ_I32(cq1, cq2);\
  I32X4X2 tm2 = VZIPQ_I32(cq3, cq4);\
  cq1 = VCOMBINE_I32(VGET_LOW_I32(tm1.val[0]), VGET_LOW_I32(tm2.val[0]));\
  cq2 = VCOMBINE_I32(VGET_HIGH_I32(tm1.val[0]), VGET_HIGH_I32(tm2.val[0]));\
  cq3 = VCOMBINE_I32(VGET_LOW_I32(tm1.val[1]), VGET_LOW_I32(tm2.val[1]));\
  cq4 = VCOMBINE_I32(VGET_HIGH_I32(tm1.val[1]), VGET_HIGH_I32(tm2.val[1]));\
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
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, bd1, bd2;\
  I32X4 cq01, cq02, cq03, cq04, cq05, cq06, cq07, cq08;\
  I32X4 cq09, cq10, cq11, cq12, cq13, cq14, cq15, cq16;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = cq07 = cq08 = VDUPQ_N_I32(0);\
  cq09 = cq10 = cq11 = cq12 = cq13 = cq14 = cq15 = cq16 = VDUPQ_N_I32(0);\
  for (; k_left > 0; k_left--) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4); a_ptr += 8;\
    bd1 = VLD1_I16(b_ptr); bd2 = VLD1_I16(b_ptr + 4); b_ptr += 8;\
    cq01 = VMLAL_LANE_I16(cq01, ad1, bd1, 0);\
    cq02 = VMLAL_LANE_I16(cq02, ad2, bd1, 0);\
    cq03 = VMLAL_LANE_I16(cq03, ad1, bd1, 1);\
    cq04 = VMLAL_LANE_I16(cq04, ad2, bd1, 1);\
    cq05 = VMLAL_LANE_I16(cq05, ad1, bd1, 2);\
    cq06 = VMLAL_LANE_I16(cq06, ad2, bd1, 2);\
    cq07 = VMLAL_LANE_I16(cq07, ad1, bd1, 3);\
    cq08 = VMLAL_LANE_I16(cq08, ad2, bd1, 3);\
    cq09 = VMLAL_LANE_I16(cq09, ad1, bd2, 0);\
    cq10 = VMLAL_LANE_I16(cq10, ad2, bd2, 0);\
    cq11 = VMLAL_LANE_I16(cq11, ad1, bd2, 1);\
    cq12 = VMLAL_LANE_I16(cq12, ad2, bd2, 1);\
    cq13 = VMLAL_LANE_I16(cq13, ad1, bd2, 2);\
    cq14 = VMLAL_LANE_I16(cq14, ad2, bd2, 2);\
    cq15 = VMLAL_LANE_I16(cq15, ad1, bd2, 3);\
    cq16 = VMLAL_LANE_I16(cq16, ad2, bd2, 3);\
  }

#define SAVE_M8N8 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M8N2(cq01, cq02, cq03, cq04)\
  UNIT_SAVE_M8N2(cq05, cq06, cq07, cq08)\
  UNIT_SAVE_M8N2(cq09, cq10, cq11, cq12)\
  UNIT_SAVE_M8N2(cq13, cq14, cq15, cq16)

#define KERNEL_M12N1_UNIT(a_head, b_head) \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, ad3;\
  I16 bs1;\
  I32X4 cq1, cq2, cq3;\
  cq1 = cq2 = cq3 = VDUPQ_N_I32(0);\
  for (; k_left > 0; k_left--) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); a_ptr += 12;\
    bs1 = *b_ptr++;\
    cq1 = VMLAL_N_I16(cq1, ad1, bs1);\
    cq2 = VMLAL_N_I16(cq2, ad2, bs1);\
    cq3 = VMLAL_N_I16(cq3, ad3, bs1);\
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
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, ad3, ad4, ad5, ad6, bd1;\
  I32X4 cq1, cq2, cq3, cq4, cq5, cq6;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = VDUPQ_N_I32(0);\
  for (; k_left > 1; k_left -= 2) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); ad4 = VLD1_I16(a_ptr + 12);\
    ad5 = VLD1_I16(a_ptr + 16); ad6 = VLD1_I16(a_ptr + 20); a_ptr += 24;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq1 = VMLAL_LANE_I16(cq1, ad1, bd1, 0);\
    cq2 = VMLAL_LANE_I16(cq2, ad2, bd1, 0);\
    cq3 = VMLAL_LANE_I16(cq3, ad3, bd1, 0);\
    cq4 = VMLAL_LANE_I16(cq4, ad1, bd1, 1);\
    cq5 = VMLAL_LANE_I16(cq5, ad2, bd1, 1);\
    cq6 = VMLAL_LANE_I16(cq6, ad3, bd1, 1);\
    cq1 = VMLAL_LANE_I16(cq1, ad4, bd1, 2);\
    cq2 = VMLAL_LANE_I16(cq2, ad5, bd1, 2);\
    cq3 = VMLAL_LANE_I16(cq3, ad6, bd1, 2);\
    cq4 = VMLAL_LANE_I16(cq4, ad4, bd1, 3);\
    cq5 = VMLAL_LANE_I16(cq5, ad5, bd1, 3);\
    cq6 = VMLAL_LANE_I16(cq6, ad6, bd1, 3);\
  }\
  if (k_left > 0) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); a_ptr += 12;\
    I16 bs1 = b_ptr[0];\
    I16 bs2 = b_ptr[1]; b_ptr += 2;\
    cq1 = VMLAL_N_I16(cq1, ad1, bs1);\
    cq2 = VMLAL_N_I16(cq2, ad2, bs1);\
    cq3 = VMLAL_N_I16(cq3, ad3, bs1);\
    cq4 = VMLAL_N_I16(cq4, ad1, bs2);\
    cq5 = VMLAL_N_I16(cq5, ad2, bs2);\
    cq6 = VMLAL_N_I16(cq6, ad3, bs2);\
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
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, ad3, bd1;\
  I32X4 cq01, cq02, cq03, cq04, cq05, cq06;\
  I32X4 cq07, cq08, cq09, cq10, cq11, cq12;\
  cq01 = cq02 = cq03 = cq04 = cq05 = cq06 = VDUPQ_N_I32(0);\
  cq07 = cq08 = cq09 = cq10 = cq11 = cq12 = VDUPQ_N_I32(0);\
  for (; k_left > 0; k_left--) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); a_ptr += 12;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq01 = VMLAL_LANE_I16(cq01, ad1, bd1, 0);\
    cq02 = VMLAL_LANE_I16(cq02, ad2, bd1, 0);\
    cq03 = VMLAL_LANE_I16(cq03, ad3, bd1, 0);\
    cq04 = VMLAL_LANE_I16(cq04, ad1, bd1, 1);\
    cq05 = VMLAL_LANE_I16(cq05, ad2, bd1, 1);\
    cq06 = VMLAL_LANE_I16(cq06, ad3, bd1, 1);\
    cq07 = VMLAL_LANE_I16(cq07, ad1, bd1, 2);\
    cq08 = VMLAL_LANE_I16(cq08, ad2, bd1, 2);\
    cq09 = VMLAL_LANE_I16(cq09, ad3, bd1, 2);\
    cq10 = VMLAL_LANE_I16(cq10, ad1, bd1, 3);\
    cq11 = VMLAL_LANE_I16(cq11, ad2, bd1, 3);\
    cq12 = VMLAL_LANE_I16(cq12, ad3, bd1, 3);\
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

#define KERNEL_M6N1_UNIT(a_head, b_head) \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, ad3, ad4, ad5, ad6, bd1;\
  I16X4X2 add1;\
  I32X4 cq1, cq2, cq3, cq4, cq5, cq6;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = VDUPQ_N_I32(0);\
  for (; k_left > 3; k_left -= 4) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); ad4 = VLD1_I16(a_ptr + 12);\
    ad5 = VLD1_I16(a_ptr + 16); ad6 = VLD1_I16(a_ptr + 20); a_ptr += 24;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    add1 = VUZP_I16(ad2, ad5);\
    cq1 = VMLAL_LANE_I16(cq1, ad1, bd1, 0);\
    cq2 = VMLAL_I16(cq2, add1.val[0], bd1);\
    cq3 = VMLAL_LANE_I16(cq3, ad3, bd1, 1);\
    cq4 = VMLAL_LANE_I16(cq4, ad4, bd1, 2);\
    cq5 = VMLAL_I16(cq5, add1.val[1], bd1);\
    cq6 = VMLAL_LANE_I16(cq6, ad6, bd1, 3);\
  }\
  cq1 = VADDQ_I32(cq1, cq4); cq3 = VADDQ_I32(cq3, cq6);\
  cq4 = VCOMBINE_I32(VGET_LOW_I32(cq2), VGET_LOW_I32(cq5));\
  cq6 = VCOMBINE_I32(VGET_HIGH_I32(cq2), VGET_HIGH_I32(cq5));\
  cq2 = VADDQ_I32(cq4, cq6);\
  I32 cs1 = VGETQ_LANE_I32(cq1, 0) + VGETQ_LANE_I32(cq2, 1);\
  I32 cs2 = VGETQ_LANE_I32(cq1, 1) + VGETQ_LANE_I32(cq2, 3);\
  I32 cs3 = VGETQ_LANE_I32(cq1, 2) + VGETQ_LANE_I32(cq3, 0);\
  I32 cs4 = VGETQ_LANE_I32(cq1, 3) + VGETQ_LANE_I32(cq3, 1);\
  I32 cs5 = VGETQ_LANE_I32(cq2, 0) + VGETQ_LANE_I32(cq3, 2);\
  I32 cs6 = VGETQ_LANE_I32(cq2, 2) + VGETQ_LANE_I32(cq3, 3);\
  for (; k_left > 0; k_left--) {\
    I32 bs1 = *b_ptr++;\
    cs1 += bs1 * (I32)a_ptr[0];\
    cs2 += bs1 * (I32)a_ptr[1];\
    cs3 += bs1 * (I32)a_ptr[2];\
    cs4 += bs1 * (I32)a_ptr[3];\
    cs5 += bs1 * (I32)a_ptr[4];\
    cs6 += bs1 * (I32)a_ptr[5];\
    a_ptr += 6;\
  }

#define KERNEL_M6N1 KERNEL_M6N1_UNIT(a_head, b_head)
#define KERNEL_M1N6 KERNEL_M6N1_UNIT(b_head, a_head)

#define SAVE_M6N1 \
  cs1 += c_ptr[0] * beta; cs2 += c_ptr[1] * beta;\
  cs3 += c_ptr[2] * beta; cs4 += c_ptr[3] * beta;\
  cs5 += c_ptr[4] * beta; cs6 += c_ptr[5] * beta;\
  c_ptr[0] = cs1; c_ptr[1] = cs2; c_ptr[2] = cs3;\
  c_ptr[3] = cs4; c_ptr[4] = cs5; c_ptr[5] = cs6;

#define SAVE_M1N6 \
  I32 *c_tmp = c_ptr;\
  cs1 += c_tmp[0] * beta; cs2 += c_tmp[ldc] * beta;\
  c_tmp[0] = cs1; c_tmp[ldc] = cs2; c_tmp += ldc * 2;\
  cs3 += c_tmp[0] * beta; cs4 += c_tmp[ldc] * beta;\
  c_tmp[0] = cs3; c_tmp[ldc] = cs4; c_tmp += ldc * 2;\
  cs5 += c_tmp[0] * beta; cs6 += c_tmp[ldc] * beta;\
  c_tmp[0] = cs5; c_tmp[ldc] = cs6;

#define KERNEL_M6N2_UNIT(a_head, b_head) \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, ad3, bd1;\
  I16X4X2 bdd1;\
  I32X4 cq1, cq2, cq3, cq4, cq5, cq6;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = VDUPQ_N_I32(0);\
  for (; k_left > 1; k_left -= 2) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); a_ptr += 12;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    bdd1 = VTRN_I16(bd1, bd1);\
    cq1 = VMLAL_LANE_I16(cq1, ad1, bd1, 0);\
    cq2 = VMLAL_I16(cq2, ad2, bdd1.val[0]);\
    cq3 = VMLAL_LANE_I16(cq3, ad3, bd1, 2);\
    cq4 = VMLAL_LANE_I16(cq4, ad1, bd1, 1);\
    cq5 = VMLAL_I16(cq5, ad2, bdd1.val[1]);\
    cq6 = VMLAL_LANE_I16(cq6, ad3, bd1, 3);\
  }\
  I32X2 cd1 = VADD_I32(VGET_LOW_I32(cq1), VGET_HIGH_I32(cq2));\
  I32X2 cd2 = VADD_I32(VGET_HIGH_I32(cq1), VGET_LOW_I32(cq3));\
  I32X2 cd3 = VADD_I32(VGET_LOW_I32(cq2), VGET_HIGH_I32(cq3));\
  I32X2 cd4 = VADD_I32(VGET_LOW_I32(cq4), VGET_HIGH_I32(cq5));\
  I32X2 cd5 = VADD_I32(VGET_HIGH_I32(cq4), VGET_LOW_I32(cq6));\
  I32X2 cd6 = VADD_I32(VGET_LOW_I32(cq5), VGET_HIGH_I32(cq6));\
  cq1 = VCOMBINE_I32(cd1, cd2); cq2 = VCOMBINE_I32(cd4, cd5);\
  I32 cs1 = VGET_LANE_I32(cd3, 0);\
  I32 cs2 = VGET_LANE_I32(cd3, 1);\
  I32 cs3 = VGET_LANE_I32(cd6, 0);\
  I32 cs4 = VGET_LANE_I32(cd6, 1);\
  if (k_left > 0) {\
    ad1 = VLD1_I16(a_ptr);\
    I32 as1 = a_ptr[4];\
    I32 as2 = a_ptr[5]; a_ptr += 6;\
    I32 bs1 = b_ptr[0];\
    I32 bs2 = b_ptr[1]; b_ptr += 2;\
    cq1 = VMLAL_N_I16(cq1, ad1, bs1);\
    cq2 = VMLAL_N_I16(cq2, ad1, bs2);\
    cs1 += as1 * bs1; cs2 += as2 * bs1;\
    cs3 += as1 * bs2; cs4 += as2 * bs2;\
  }

#define KERNEL_M6N2 KERNEL_M6N2_UNIT(a_head, b_head)
#define KERNEL_M2N6 KERNEL_M6N2_UNIT(b_head, a_head)

#define SAVE_M6N2 \
  I32 *c_l1 = c_ptr + ldc;\
  cq1 = VMLAQ_N_I32(cq1, VLD1Q_I32(c_ptr), beta);\
  cq2 = VMLAQ_N_I32(cq2, VLD1Q_I32(c_l1), beta);\
  cs1 += c_ptr[4] * beta; cs2 += c_ptr[5] * beta;\
  cs3 += c_l1[4] * beta; cs4 += c_l1[5] * beta;\
  VST1Q_I32(c_ptr, cq1); VST1Q_I32(c_l1, cq2);\
  c_ptr[4] = cs1; c_ptr[5] = cs2;\
  c_l1[4] = cs3; c_l1[5] = cs4;

#define SAVE_M2N6 \
  I32 *c_tmp = c_ptr; UNIT_SAVE_M2N4(cq1, cq2)\
  cs1 += c_tmp[0] * beta; cs3 += c_tmp[1] * beta;\
  c_tmp[0] = cs1; c_tmp[1] = cs3; c_tmp += ldc;\
  cs2 += c_tmp[0] * beta; cs4 += c_tmp[1] * beta;\
  c_tmp[0] = cs2; c_tmp[1] = cs4;

#define KERNEL_M6N4_UNIT(a_head, b_head) \
  COMMON_KERNEL_HEADER(a_head, b_head) \
  I16X4 ad1, ad2, ad3, bd1, bd2;\
  I32X4 cq1, cq2, cq3, cq4, cq5, cq6;\
  cq1 = cq2 = cq3 = cq4 = cq5 = cq6 = VDUPQ_N_I32(0);\
  for (; k_left > 1; k_left -= 2) {\
    ad1 = VLD1_I16(a_ptr); ad2 = VLD1_I16(a_ptr + 4);\
    ad3 = VLD1_I16(a_ptr + 8); a_ptr += 12;\
    bd1 = VLD1_I16(b_ptr); bd2 = VLD1_I16(b_ptr + 4); b_ptr += 8;\
    cq1 = VMLAL_LANE_I16(cq1, bd1, ad1, 0);\
    cq2 = VMLAL_LANE_I16(cq2, bd1, ad1, 1);\
    cq3 = VMLAL_LANE_I16(cq3, bd1, ad1, 2);\
    cq4 = VMLAL_LANE_I16(cq4, bd1, ad1, 3);\
    cq5 = VMLAL_LANE_I16(cq5, bd1, ad2, 0);\
    cq6 = VMLAL_LANE_I16(cq6, bd1, ad2, 1);\
    cq1 = VMLAL_LANE_I16(cq1, bd2, ad2, 2);\
    cq2 = VMLAL_LANE_I16(cq2, bd2, ad2, 3);\
    cq3 = VMLAL_LANE_I16(cq3, bd2, ad3, 0);\
    cq4 = VMLAL_LANE_I16(cq4, bd2, ad3, 1);\
    cq5 = VMLAL_LANE_I16(cq5, bd2, ad3, 2);\
    cq6 = VMLAL_LANE_I16(cq6, bd2, ad3, 3);\
  }\
  if (k_left > 0) {\
    ad1 = VLD1_I16(a_ptr);\
    I32 as1 = a_ptr[4];\
    I32 as2 = a_ptr[5]; a_ptr += 6;\
    bd1 = VLD1_I16(b_ptr); b_ptr += 4;\
    cq1 = VMLAL_LANE_I16(cq1, bd1, ad1, 0);\
    cq2 = VMLAL_LANE_I16(cq2, bd1, ad1, 1);\
    cq3 = VMLAL_LANE_I16(cq3, bd1, ad1, 2);\
    cq4 = VMLAL_LANE_I16(cq4, bd1, ad1, 3);\
    cq5 = VMLAL_N_I16(cq5, bd1, as1);\
    cq6 = VMLAL_N_I16(cq6, bd1, as2);\
  }

#define KERNEL_M6N4 KERNEL_M6N4_UNIT(a_head, b_head)
#define KERNEL_M4N6 KERNEL_M6N4_UNIT(b_head, a_head)

#define UNIT_SAVE_M6N4(cq1, cq2, cq3, cq4, cq5, cq6) \
  UNIT_SAVE_M4N4_TRANS(cq1, cq2, cq3, cq4)\
  c_tmp -= 4 * ldc;\
  c_tmp += 4;\
  UNIT_SAVE_M2N4(cq5, cq6)\
  c_tmp -= 4;

#define SAVE_M6N4 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M6N4(cq1, cq2, cq3, cq4, cq5, cq6)

#define SAVE_M4N6 \
  I32 *c_tmp = c_ptr;\
  UNIT_SAVE_M4N2(cq1, cq2) UNIT_SAVE_M4N2(cq3, cq4) UNIT_SAVE_M4N2(cq5, cq6)

#define NEON_IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(mdim, ndim, srcint, dstint) \
static inline void\
  inline_dualpack_gemm_a##srcint##_b##srcint##_c##dstint##_m##mdim##_n##ndim(\
  const srcint *a_head, const srcint *b_head, dstint *c_ptr,\
  uint32_t K, dstint beta, uint32_t ldc) {\
  KERNEL_M##mdim##N##ndim\
  SAVE_M##mdim##N##ndim\
}

#define IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(mdim, ndim, srcint, dstint)\
  NEON_IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(mdim, ndim, srcint, dstint)

IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 1, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 2, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 1, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 2, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 4, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 4, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 1, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 2, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 4, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 8, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 8, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 8, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 1, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 2, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 4, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(8, 8, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 6, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 6, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 6, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(6, 1, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(6, 2, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(6, 4, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(1, 12, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(2, 12, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(4, 12, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 1, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 2, I16, I32)
IMLAGEMM_INLINE_DUALPACK_UNIT_FUNC(12, 4, I16, I32)

#endif
