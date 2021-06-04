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
 * File:        NeonExtreme.h
 * Description: Source code template for NEON max/min functions.
 *****************************************************************************/

#include "common/ExpandMacro.h"
#include <stdint.h>
#include <arm_neon.h>

#ifndef INCLUDE_NEON_EXTREME
#define INCLUDE_NEON_EXTREME

#define NEON_REDUC_S_ITEM(n, type, short) {\
  type tmin = vget_lane_##short(vmin1d, n - 1);\
  type tmax = vget_lane_##short(vmax1d, n - 1);\
  smin = tmin < smin ? tmin : smin;\
  smax = tmax > smax ? tmax : smax;\
}

#define NEON_REDUC_S_MIN_MAX(n, type, short) \
  MACRO_EXPANSION_##n(VOID_BASE, NEON_REDUC_S_ITEM, type, short)

#define NEON_FIND_EXTREME(type, short, dvec, qvec, dlen) \
static inline void inline_find_extreme_##type(const type *dat, uint32_t size,\
  type *min, type *max) {\
\
  qvec vmin1, vmin2, vmin3, vmin4;\
  qvec vmax1, vmax2, vmax3, vmax4;\
\
  if (size == 0) return;\
  vmin1 = vmin2 = vmin3 = vmin4 = \
    vmax1 = vmax2 = vmax3 = vmax4 = vld1q_dup_##short(dat);\
  uint32_t elem_left = size;\
  for (; elem_left >= dlen * 8; elem_left -= dlen * 8) {\
    qvec l1 = vld1q_##short(dat);\
    qvec l2 = vld1q_##short(dat + dlen * 2);\
    qvec l3 = vld1q_##short(dat + dlen * 4);\
    qvec l4 = vld1q_##short(dat + dlen * 6);\
    dat += dlen * 8;\
    vmin1 = vminq_##short(vmin1, l1);\
    vmax1 = vmaxq_##short(vmax1, l1);\
    vmin2 = vminq_##short(vmin2, l2);\
    vmax2 = vmaxq_##short(vmax2, l2);\
    vmin3 = vminq_##short(vmin3, l3);\
    vmax3 = vmaxq_##short(vmax3, l3);\
    vmin4 = vminq_##short(vmin4, l4);\
    vmax4 = vmaxq_##short(vmax4, l4);\
  }\
  vmin1 = vminq_##short(vmin1, vmin3);\
  vmin2 = vminq_##short(vmin2, vmin4);\
  vmax1 = vmaxq_##short(vmax1, vmax3);\
  vmax2 = vmaxq_##short(vmax2, vmax4);\
  if (elem_left >= dlen * 4) {\
    qvec l1 = vld1q_##short(dat);\
    qvec l2 = vld1q_##short(dat + dlen * 2);\
    dat += dlen * 4;\
    vmin1 = vminq_##short(vmin1, l1);\
    vmax1 = vmaxq_##short(vmax1, l1);\
    vmin2 = vminq_##short(vmin2, l2);\
    vmax2 = vmaxq_##short(vmax2, l2);\
    elem_left -= dlen * 4;\
  }\
  vmin1 = vminq_##short(vmin1, vmin2);\
  vmax1 = vmaxq_##short(vmax1, vmax2);\
  if (elem_left >= dlen * 2) {\
    qvec l1 = vld1q_##short(dat);\
    dat += dlen * 2;\
    vmin1 = vminq_##short(vmin1, l1);\
    vmax1 = vmaxq_##short(vmax1, l1);\
    elem_left -= dlen * 2;\
  }\
  dvec vmin1d = vmin_##short(vget_low_##short(vmin1),\
    vget_high_##short(vmin1));\
  dvec vmax1d = vmax_##short(vget_low_##short(vmax1),\
    vget_high_##short(vmax1));\
  if (elem_left >= dlen) {\
    dvec d1 = vld1_##short(dat);\
    dat += dlen;\
    vmin1d = vmin_##short(vmin1d, d1);\
    vmax1d = vmax_##short(vmax1d, d1);\
    elem_left -= dlen;\
  }\
  type smin = vget_lane_##short(vmin1d, 0);\
  type smax = vget_lane_##short(vmax1d, 0);\
  NEON_REDUC_S_MIN_MAX(dlen, type, short)\
  for (; elem_left > 0; elem_left--) {\
    type s1 = *dat++;\
    smin = s1 < smin ? s1 : smin;\
    smax = s1 > smax ? s1 : smax;\
  }\
  *min = smin;\
  *max = smax;\
}

#endif
