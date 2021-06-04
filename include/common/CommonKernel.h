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
 * File:        CommonKernel.h
 * Description: The common skeleton of regular GEMM kernel functions with both
 *              source matrices packed before computation
 * Extention:   For supporting a new CPU arch, the following steps are needed
 *              in addition to including this header:
 *              (1) implement a collection of inline GEMM functions, each with
 *                  fixed M & N but variable K(as input param), for the
 *                  multiplication of column-major matrix A with row-major
 *                  matrix B and update the results to column-major matrix C.
 *                  A SGEMM inline function with M = 2 and N = 4 can be
 *                  implemented like this:
 *                   static inline void
 *                     inline_dualpack_gemm_afloat_bfloat_cfloat_m2_n4(
 *                       const float *a_head, const float *b_head,
 *                         float *c_ptr, uint32_t K, float beta, uint32_t ldc) {
 *                           float c0, c1, c2, c3, c4, c5, c6, c7;
 *                           c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0.0f;
 *                           for (; K > 0; K--) {
 *                             float a0 = a_head[0];
 *                             float a1 = a_head[1]; a_head += 2;
 *                             float b0 = b_head[0];
 *                             float b1 = b_head[1];
 *                             float b2 = b_head[2];
 *                             float b3 = b_head[3]; b_head += 4;
 *                             c0 += a0 * b0; c1 += a1 * b0;
 *                             c2 += a0 * b1; c3 += a1 * b1;
 *                             c4 += a0 * b2; c5 += a1 * b2;
 *                             c6 += a0 * b3; c7 += a1 * b3;
 *                           }
 *                           c_ptr[0] = c_ptr[0] * beta + c0;
 *                           c_ptr[1] = c_ptr[1] * beta + c1;
 *                           c_ptr += ldc;
 *                           c_ptr[0] = c_ptr[0] * beta + c2;
 *                           c_ptr[1] = c_ptr[1] * beta + c3;
 *                           c_ptr += ldc;
 *                           c_ptr[0] = c_ptr[0] * beta + c4;
 *                           c_ptr[1] = c_ptr[1] * beta + c5;
 *                           c_ptr += ldc;
 *                           c_ptr[0] = c_ptr[0] * beta + c6;
 *                           c_ptr[1] = c_ptr[1] * beta + c7;
 *                         }
 *              (2) Construct kernel functions with the aid of macros.
 *                  Please refer to src/neon_armv7a/SgemmKernel.c for example.
 *****************************************************************************/

#include "ExpandMacro.h"
#include <stdint.h>

#ifndef INCLUDE_COMMON_KERNEL
#define INCLUDE_COMMON_KERNEL

/* the macros COMPUTE_MmNn are architecture dependant,
 * which should be defined in the source file including this header */

#define COMPUTE_STD_INIT_SLICE(n_pos, mdim, ctype) \
  ctype c_reg##n_pos[mdim];\
  _Pragma("omp simd")\
  for (int j = 0; j < mdim; ++j) {\
    c_reg##n_pos[j] = 0;\
  }

#define COMPUTE_STD_ACC_SLICE(n_pos, mdim, ndim, k_off) \
  _Pragma("omp simd")\
  for (int j = 0; j < mdim; ++j) {\
    c_reg##n_pos[j] += a_ptr[j + k_off * mdim] *\
      b_ptr[n_pos - 1 + k_off * ndim];\
  }

#define COMPUTE_STD_SAVE_SLICE(n_pos, mdim, c_str) \
  _Pragma("omp simd")\
  for (int j = 0; j < mdim; ++j) {\
    c_str[j] = c_str[j] * beta + c_reg##n_pos[j];\
  }\
  c_str += ldc;

#define COMPUTE_STD(mdim, ndim, atype, btype, ctype) \
static inline void\
  inline_dualpack_gemm_a##atype##_b##btype##_c##ctype##_m##mdim##_n##ndim(\
  const atype *a_head, const btype *b_head, ctype *c_ptr,\
  uint32_t K, ctype beta, uint32_t ldc) {\
  MACRO_EXP_##ndim(VOID_BASE, COMPUTE_STD_INIT_SLICE, mdim, ctype)\
  const atype * a_ptr = a_head;\
  const btype * b_ptr = b_head;\
  uint32_t k_left = K;\
  for (; k_left > 3; k_left -= 4) {\
    MACRO_EXP_##ndim(VOID_BASE, COMPUTE_STD_ACC_SLICE, mdim, ndim, 0)\
    MACRO_EXP_##ndim(VOID_BASE, COMPUTE_STD_ACC_SLICE, mdim, ndim, 1)\
    MACRO_EXP_##ndim(VOID_BASE, COMPUTE_STD_ACC_SLICE, mdim, ndim, 2)\
    MACRO_EXP_##ndim(VOID_BASE, COMPUTE_STD_ACC_SLICE, mdim, ndim, 3)\
    a_ptr += mdim * 4;\
    b_ptr += ndim * 4;\
  }\
  for (; k_left > 0; k_left--) {\
    MACRO_EXP_##ndim(VOID_BASE, COMPUTE_STD_ACC_SLICE, mdim, ndim, 0)\
    a_ptr += mdim;\
    b_ptr += ndim;\
  }\
  ctype *c_str = c_ptr;\
  MACRO_EXP_##ndim(VOID_BASE, COMPUTE_STD_SAVE_SLICE, mdim, c_str)\
}

#define MICRO_COMPUTE_LM_LOOP(mdim, ndim, atype, btype, ctype) \
  for (; m_left >= mdim; m_left -= mdim) {\
    inline_dualpack_gemm_a##atype##_b##btype##_c##ctype##_m##mdim##_n##ndim(\
      a_head, b_head, c_ptr, K, beta, ldc);\
    a_head += mdim * K;\
    c_ptr += mdim;\
  }

#define MICRO_COMPUTE_LN_LOOP(ndim, mdim, atype, btype, ctype) \
  for (; n_left >= ndim; n_left -= ndim) {\
    inline_dualpack_gemm_a##atype##_b##btype##_c##ctype##_m##mdim##_n##ndim(\
      a_head, b_head, c_ptr, K, beta, ldc);\
    b_head += ndim * K;\
    c_ptr += ndim * ldc;\
  }

#define MICRO_COMPUTE_LM(mdim, ndim, atype, btype, ctype) \
  MACRO_EXPANSION_E_##mdim(MICRO_COMPUTE_LM_LOOP, ndim, atype, btype, ctype)

#define MICRO_COMPUTE_LN(mdim, ndim, atype, btype, ctype) \
  MACRO_EXPANSION_E_##ndim(MICRO_COMPUTE_LN_LOOP, mdim, atype, btype, ctype)

#define DUALPACK_COMPUTE_LM(ndim, satype, sbtype, ctype, block_m_max) \
  for (; n_left >= ndim; n_left -= ndim) {\
    const satype *a_head = sa;\
    ctype *c_ptr = c_head;\
    uint32_t m_left = M;\
    MICRO_COMPUTE_LM(block_m_max, ndim, satype, sbtype, ctype)\
    b_head += K * ndim;\
    c_head += ldc * ndim;\
  }

#define DUALPACK_COMPUTE_LN(mdim, satype, sbtype, ctype, block_n_max) \
  for (; m_left >= mdim; m_left -= mdim) {\
    const sbtype *b_head = sb;\
    ctype *c_ptr = c_head;\
    uint32_t n_left = N;\
    MICRO_COMPUTE_LN(mdim, block_n_max, satype, sbtype, ctype)\
    a_head += K * mdim;\
    c_head += mdim;\
  }

#define ASSEMBLE_DUALPACK_COMPUTE_LM(ndim, satype, sbtype, ctype, block_m_max) \
  MACRO_EXP_E_##ndim(DUALPACK_COMPUTE_LM, satype, sbtype, ctype, block_m_max)

#define ASSEMBLE_DUALPACK_COMPUTE_LN(mdim, satype, sbtype, ctype, block_n_max) \
  MACRO_EXP_E_##mdim(DUALPACK_COMPUTE_LN, satype, sbtype, ctype, block_n_max)

#define DUALPACK_KERNEL_FUNC_LM(gemmtype, satype, sbtype, ctype, block_m_max, block_n_max) \
void gemmtype##_kernel_lm_m##block_m_max##n##block_n_max(\
  uint32_t M, uint32_t N, uint32_t K, ctype beta,\
  const satype * __restrict__ sa, const sbtype * __restrict__ sb,\
  ctype * __restrict__ C, uint32_t ldc) {\
  uint32_t n_left = N;\
  const sbtype *b_head = sb;\
  ctype *c_head = C;\
  ASSEMBLE_DUALPACK_COMPUTE_LM(block_n_max, satype, sbtype, ctype, block_m_max)\
}

#define DUALPACK_KERNEL_FUNC_LN(gemmtype, satype, sbtype, ctype, block_m_max, block_n_max) \
void gemmtype##_kernel_ln_m##block_m_max##n##block_n_max(\
  uint32_t M, uint32_t N, uint32_t K, ctype beta,\
  const satype * __restrict__ sa, const sbtype * __restrict__ sb,\
  ctype * __restrict__ C, uint32_t ldc) {\
  uint32_t m_left = M;\
  const satype *a_head = sa;\
  ctype *c_head = C;\
  ASSEMBLE_DUALPACK_COMPUTE_LN(block_m_max, satype, sbtype, ctype, block_n_max)\
}

#endif
