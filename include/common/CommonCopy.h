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
 * File:        CommonCopy.h
 * Description: Common building blocks for packing functions in GEMM operation
 * Terms:       "ncopy": pack from K-major source matrix
 *              "tcopy": pack from K-minor source matrix
 *****************************************************************************/

#include "ExpandMacro.h"
#include <stdint.h>

#ifndef INCLUDE_COMMON_COPY
#define INCLUDE_COMMON_COPY

#define NCOPY_INIT_SRC_PTR_ITEM(n, type) \
  const type *src##n = src0 + (n - 1) * ld_dim;
#define NCOPY_INIT_SRC_PTR(n, type) \
  MACRO_EXPANSION_##n(VOID_BASE, NCOPY_INIT_SRC_PTR_ITEM, type)

#define NCOPY_COPY_1(n) \
  dst1[n - 1] = *src##n; src##n ++;
#define NCOPY_COPY(n) \
  MACRO_EXPANSION_##n(VOID_BASE, NCOPY_COPY_1) dst1 += n;

/* a standard-C fallback for NCOPY_<type_<stype> */
#define NCOPY_STD(unroll) \
  for (; dim1_count > 0; dim1_count--) {\
    NCOPY_COPY(unroll)\
  }

/* the macro NCOPY_<type>_<stype>(unroll) is architecture dependant,
 *  * which should be defined in the source file including this header */
#define NCOPY_LOOP(unroll, type, stype) \
  for (; dim2_count >= unroll; dim2_count -= unroll) {\
    uint32_t dim1_count = dim1;\
    NCOPY_INIT_SRC_PTR(unroll, type)\
    NCOPY_##type##_##stype(unroll)\
    src0 += ld_dim * unroll;\
  }
#define NCOPY(max_unroll, side, type) \
  MACRO_EXP_E_##max_unroll(NCOPY_LOOP, side, type)

#define GENERIC_NCOPY_FUNC(gemmtype, type, stype, max_unroll) \
void gemmtype##_##type##_##stype##_ncopy_unroll##max_unroll(\
  const type * __restrict__ src, stype * __restrict__ dst,\
  uint32_t ld_dim, uint32_t dim1, uint32_t dim2) {\
  const type *src0 = src;\
  stype *dst1 = dst;\
  uint32_t dim2_count = dim2;\
  NCOPY(max_unroll, type, stype)\
}


/* this macro is the fallback for TCOPY_UNIT_<type>_<stype> */
#define TCOPY_UNIT_STD(src_ptr, dst_ptr, dst_offset, num_elements) \
  _Pragma("omp simd")\
  for (int i = 0; i < num_elements; ++i) \
    dst_ptr[dst_offset + i] = src_ptr[i];

/* the macro 
 * TCOPY_UNIT_<type>_<stype>(src_ptr, dst_ptr, dst_offset, num_elements)
 * is architecture dependant,
 * which should be defined in source file including this header */

#define TCOPY_LINE_1(n, unroll, type, stype) \
  TCOPY_UNIT_##type##_##stype(src##n, dst1, ((n-1)*unroll), unroll)\
  src##n += unroll;
#define TCOPY_LINES(n, unroll, type, stype) \
  MACRO_EXPANSION_##n(VOID_BASE, TCOPY_LINE_1, unroll, type, stype)

#define TCOPY_LOOP(unroll, type, stype, read_width) \
  dst1 = dst + (dim1 - dim1_count) * dim2 + (dim2 - dim2_count) * unroll;\
  for (; dim1_count >= unroll; dim1_count -= unroll) {\
    TCOPY_LINES(read_width, unroll, type, stype)\
    dst1 += dim2 * unroll;\
  }
#define TCOPY(max_unroll, type, stype, read_width) \
  MACRO_EXPANSION_E_##max_unroll(TCOPY_LOOP, type, stype, read_width)

#define GENERIC_TCOPY_FUNC(gemmtype, type, stype, max_unroll) \
void gemmtype##_##type##_##stype##_tcopy_unroll##max_unroll(\
  const type * __restrict__ src, stype * __restrict__ dst,\
  uint32_t ld_dim, uint32_t dim1, uint32_t dim2) {\
  uint32_t dim2_count = dim2;\
  const type *src0 = src;\
  for (; dim2_count > 3; dim2_count -= 4) {\
    const type *src1 = src0;\
    const type *src2 = src0 + ld_dim;\
    const type *src3 = src0 + ld_dim * 2;\
    const type *src4 = src2 + ld_dim * 2;\
    stype *dst1;\
    uint32_t dim1_count = dim1;\
    TCOPY(max_unroll, type, stype, 4)\
    src0 += ld_dim * 4;\
  }\
  for (; dim2_count > 0; dim2_count--) {\
    const type *src1 = src0;\
    stype *dst1;\
    uint32_t dim1_count = dim1;\
    TCOPY(max_unroll, type, stype, 1)\
    src0 += ld_dim;\
  }\
}

#endif
