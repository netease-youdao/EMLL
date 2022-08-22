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
* File:        CommonSkinnyGer.h
* Description: Common building blocks for regular * skinny or skinny * regular
*              matmul when the regular matrix is column-major in the former
*              case or row-major in the latter case. These 2 kinds of matmul
*              involving skinny matrices require a special efficient kernel
*              different from that in regular * regular matmul. Specifically,
*              The regular matrix is no longer reordered (packed) during
*              calculation. Elements from the regular matrix are accessed
*              sequentially and only once.The GEMM calculation is decomposed
*              into sequential GER operations rather than DOT ones.The
*              output matrix is accessed repeatedly in GER operations so
*              it is always packed in a scratch array.
* Extension:   To support a new CPU architecture, the following tasks should
*              be done in addition to including this header:
*              (1) Use typedef to define <gemm>_skinnyger_[a/b/c]scalar and
*                  <gemm>_skinnyger_[a/b/c]vec[\d]. For example, when
*                  developing avx2 SGEMM regular*skinny kernels, the following
*                  lines should be added when the maximum vector length is 8
*                  in M and 4 in K:
*                    typedef float sgemm_skinnyger_ascalar;
*                    typedef float sgemm_skinnyger_bscalar;
*                    typedef float sgemm_skinnyger_cscalar;
*                    // M vec length up to 8
*                    typedef float sgemm_skinnyger_avec1;
*                    typedef __m128 sgemm_skinnyger_avec4;
*                    typedef __m256 sgemm_skinnyger_avec8;
*                    typedef float sgemm_skinnyger_cvec1;
*                    typedef __m128 sgemm_skinnyger_cvec4;
*                    typedef __m256 sgemm_skinnyger_cvec8;
*                    // K vec length up to 4
*                    typedef float sgemm_skinnyger_bvec1;
*                    typedef __m128 sgemm_skinnyger_bvec4;
*              (2) Implement inline functions for basic vector-scalar
*                  multiply-add operations. Here is an example for
*                  an inline function of avx2 SGEMM with
*                  m_veclen = 8, k_veclen = 4 and k_laneid = 3,
*                  which multiplies each element in a_vec with the
*                  element at lane 3 in b_vec and add the result
*                  to the corresponding element in c_vec:
*                    GEMM_SKINNY_GER_CALC_UNIT(sgemm, 8, 4, 3) {
*                      __m256 b_v0 = _mm256_broadcast_ss((float*)&b_vec + 2);
*                      return _mm256_fmadd_ps(a_vec, b_v0, c_vec);
*                    }
*                  For every combination of m_veclen and k_veclen,
*                  all related inline multiply-add functions
*                  with k_laneid from 1 to k_veclen should be implemented.
*              (3) Implement load and store inline functions for matrix
*                  a/b/c like this (each catagory 1 example):
*                    // the 3 types of functions below should be written
*                    // for each m_veclen
*                    GEMM_SKINNY_GER_LOADA_UNIT(sgemm, 8) {
*                      _mm_prefetch((char *)(a_ptr + 24), _MM_HINT_T0);
*                      return _mm256_loadu_ps(a_ptr);
*                    }
*                    GEMM_SKINNY_GER_LOADC_UNIT(sgemm, 8) {
*                      return _mm256_loadu_ps(c_ptr);
*                    }
*                    GEMM_SKINNY_GER_STOREC_UNIT(sgemm, 8) {
*                      _mm256_storeu_ps(c_ptr, c_vec);
*                    }
*                    // the 2 types of functions blow should be written
*                    // for each k_veclen
*                    GEMM_SKINNY_GER_LOADB_UNIT_BROWMAJOR(sgemm, 4) {
*                      float e0 = *b_ptr; b_ptr += ldb;
*                      float e1 = *b_ptr; b_ptr += ldb;
*                      float e2 = *b_ptr; b_ptr += ldb;
*                      float e3 = *b_ptr;
*                      return _mm_set_ps(e3, e2, e1, e0);
*                    }
*                    GEMM_SKINNY_GER_LOADB_UNIT_BCOLMAJOR(sgemm, 4) {
*                      return _mm_loadu_ps(b_ptr);
*                    }
*              (4) Finally build kernel functions from inline functions
*                  defined above. For each kernel function only 1 line
*                  is needed. The following line defines regular*skinny
*                  kernel functions (serial and OpenMP) for the minimum
*                  dimension length = 3 with m_veclen = {1, 4, 8}
*                  and k_veclen = {1, 4}:
*                    GEMM_SKINNY_GER_PARALLEL_FUNC(sgemm, 3, 5, 13, 8192,
*                      float, float)
*                  The last 2 parameters in the macro are for function
*                  name mangling, providing the data type for regular
*                  and skinny matrix respectively. The last number in
*                  macro parameters (8192) specify the scratch size
*                  for output matrix which should be adjusted to the size
*                  of L1 cache. The second number (5) is the sum of all
*                  implemented k_veclen values. The third number (13) is
*                  the sum of all m_veclen values implemented.
******************************************************************************/

#define D_SCALE 2 //dynamic scaling factor in scheduling
#include "common/ExpandMacro.h"
#include "common/CommonSched.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#ifndef EMLL_SERIAL_ONLY
#include <omp.h>
#endif

#ifndef INCLUDE_COMMON_SKINNY_GER
#define INCLUDE_COMMON_SKINNY_GER

/* GEMM_SKINNY_GER_XXX_UNIT: computation units basic in skinny_ger function */
/* below are only headers to this 6 functions */
/* the function bodies should be provided according to CPU arch */

#define GEMM_SKINNY_GER_CALC_UNIT(gemm, m_vlen, k_vlen, k_id) \
static inline gemm##_skinnyger_cvec##m_vlen\
  inline_##gemm##_acolmajor_bskinny_fma_unit_m##m_vlen##_kid##k_id##in##k_vlen(\
    gemm##_skinnyger_cvec##m_vlen c_vec,\
    gemm##_skinnyger_avec##m_vlen a_vec,\
    gemm##_skinnyger_bvec##k_vlen b_vec)
/* you should give vectorized implementation equivalent to this:
 * GEMM_SKINNY_GER_CALC_UNIT(gemm, m_vlen, k_vlen, k_id) {
 *   gemm##_skinnyger_cvec##m_vlen ret;
 *   for (int i = 0; i < m_vlen; ++i) {
 *     ret[i] = c_vec[i] + a_vec[i] * b_vec[k_id - 1];
 *   } 
 *   return ret;
 * }
 */

#define GEMM_SKINNY_GER_LOADA_UNIT(gemm, m_vlen) \
static inline gemm##_skinnyger_avec##m_vlen\
  inline_##gemm##_acolmajor_bskinny_loada_unit_m##m_vlen(\
    const gemm##_skinnyger_ascalar *a_ptr)
/* you should give vectorized implementation equivalent to this:
 * GEMM_SKINNY_GER_LOADA_UNIT(gemm, m_vlen) {
 *   gemm##_skinnyger_avec##m_vlen ret;
 *   for (int i = 0; i < m_vlen; ++i) {
 *     ret[i] = a_ptr[i];
 *   }
 *   prefetch(a_ptr + pref_distance);
 *   return ret;
 * }
 */

#define GEMM_SKINNY_GER_LOADC_UNIT(gemm, m_vlen) \
static inline gemm##_skinnyger_cvec##m_vlen\
  inline_##gemm##_acolmajor_bskinny_loadc_unit_m##m_vlen(\
    const gemm##_skinnyger_cscalar *c_ptr)
/* you should give vectorized implementation equivalent to this:
 * GEMM_SKINNY_GER_LOADC_UNIT(gemm, m_vlen) {
 *   gemm##_skinnyger_cvec##m_vlen ret;
 *   for (int i = 0; i < m_vlen; ++i) {
 *     ret[i] = c_ptr[i];
 *   }
 *   return ret;
 * }
 */

#define GEMM_SKINNY_GER_STOREC_UNIT(gemm, m_vlen) \
static inline void\
  inline_##gemm##_acolmajor_bskinny_storec_unit_m##m_vlen(\
    gemm##_skinnyger_cscalar *c_ptr,\
    gemm##_skinnyger_cvec##m_vlen c_vec)
/* you should give vectorized implementation equivalent to this:
 * GEMM_SKINNY_GER_STOREC_UNIT(gemm, m_vlen) {
 *   for (int i = 0; i < m_vlen; ++i) {
 *     c_ptr[i] = c_vec[i];
 *   }
 * }
 */

#define GEMM_SKINNY_GER_LOADB_UNIT_BROWMAJOR(gemm, k_vlen) \
static inline gemm##_skinnyger_bvec##k_vlen\
  inline_##gemm##_acolmajor_bskinny_loadb_browmajor_unit_k##k_vlen(\
    const gemm##_skinnyger_bscalar *b_ptr, uint32_t ldb)
/* you should give optimized implementation equivalent to this:
 * GEMM_SKINNY_GER_LOADB_UNIT_BROWMAJOR(gemm, k_vlen) {
 *   gemm##_skinnyger_bvec##k_vlen ret;
 *   for (int i = 0; i < m_vlen; ++i) {
 *     ret[i] = *b_ptr; b_ptr += ldb;
 *   }
 * }
 */

#define GEMM_SKINNY_GER_LOADB_UNIT_BCOLMAJOR(gemm, k_vlen) \
static inline gemm##_skinnyger_bvec##k_vlen\
  inline_##gemm##_acolmajor_bskinny_loadb_bcolmajor_unit_k##k_vlen(\
    const gemm##_skinnyger_bscalar *b_ptr)
/* you should give vectorized implementation equivalent to this:
 * GEMM_SKINNY_GER_LOADB_UNIT_BCOLMAJOR(gemm, k_vlen) {
 *   gemm##_skinnyger_bvec##k_vlen ret;
 *   for (int i = 0; i < m_vlen; ++i) {
 *     ret[i] = b_ptr[i];
 *   }
 * }
 */


/* construct skinny_ger function from computation units */
#define GEMM_SKINNY_GER_CALC_UNIT_ITEM(n_id, gemm, m_vlen, k_vlen, k_id) \
  c##m_vlen##_##n_id =\
    inline_##gemm##_acolmajor_bskinny_fma_unit_m##m_vlen##_kid##k_id##in##k_vlen(\
      c##m_vlen##_##n_id, a##m_vlen##_##k_id, b##k_vlen##_##n_id);

#define GEMM_SKINNY_GER_CALC_UNIT_K1(k_id, gemm, m_vlen, k_vlen, n_dim) \
  const gemm##_skinnyger_avec##m_vlen a##m_vlen##_##k_id =\
    inline_##gemm##_acolmajor_bskinny_loada_unit_m##m_vlen(a_ptr##k_id);\
  a_ptr##k_id += m_vlen;\
  MACRO_EXPANSION_##n_dim(VOID_BASE, GEMM_SKINNY_GER_CALC_UNIT_ITEM,\
    gemm, m_vlen, k_vlen, k_id)

#define GEMM_SKINNY_GER_LOADC_ITEM(n_id, gemm, m_vlen) \
  gemm##_skinnyger_cvec##m_vlen c##m_vlen##_##n_id =\
    inline_##gemm##_acolmajor_bskinny_loadc_unit_m##m_vlen(\
      c_ptr + (n_id - 1) * m_vlen);

#define GEMM_SKINNY_GER_STOREC_ITEM(n_id, gemm, m_vlen) \
  inline_##gemm##_acolmajor_bskinny_storec_unit_m##m_vlen(\
    c_ptr + (n_id - 1) * m_vlen, c##m_vlen##_##n_id);

#define GEMM_SKINNY_GER_COMPUTE_BLOCK(gemm, m_vlen, k_vlen, n_dim) \
  MACRO_EXPANSION_##n_dim(VOID_BASE,\
    GEMM_SKINNY_GER_LOADC_ITEM, gemm, m_vlen)\
  MACRO_EXP_##k_vlen(VOID_BASE,\
    GEMM_SKINNY_GER_CALC_UNIT_K1, gemm, m_vlen, k_vlen, n_dim)\
  MACRO_EXPANSION_##n_dim(VOID_BASE,\
    GEMM_SKINNY_GER_STOREC_ITEM, gemm, m_vlen)

#define GEMM_SKINNY_GER_COMPUTE_BLOCK_LOOP(\
  m_vlen, gemm, k_vlen, n_dim) \
  for (; m_left >= m_vlen; m_left -= m_vlen) {\
    GEMM_SKINNY_GER_COMPUTE_BLOCK(gemm, m_vlen, k_vlen, n_dim) \
    c_ptr += n_dim * m_vlen;\
  }

#define GEMM_SKINNY_GER_DECLARE_B_ITEM(n_id, gemm, k_vlen) \
  gemm##_skinnyger_bvec##k_vlen b##k_vlen##_##n_id;

#define GEMM_SKINNY_GER_LOADB_BROWMAJOR_ITEM(n_id, gemm, k_vlen) \
  b##k_vlen##_##n_id =\
    inline_##gemm##_acolmajor_bskinny_loadb_browmajor_unit_k##k_vlen(\
      b_ptr, LDB);  b_ptr++;

#define GEMM_SKINNY_GER_LOADB_BCOLMAJOR_ITEM(n_id, gemm, k_vlen) \
  b##k_vlen##_##n_id =\
    inline_##gemm##_acolmajor_bskinny_loadb_bcolmajor_unit_k##k_vlen(b_ptr);\
  b_ptr += LDB;

#define GEMM_SKINNY_GER_INIT_APTR_ITEM(k_id, gemm) \
  const gemm##_skinnyger_ascalar *a_ptr##k_id = a_ptr + (k_id - 1) * LDA;

/* define valid inline function */
#define GEMM_SKINNY_GER_INLINE_FUNC(gemm, n_dim, k_vlen, m_mask) \
static inline void inline_##gemm##_acolmajor_bskinny_k##k_vlen##n##n_dim(\
  const gemm##_skinnyger_ascalar *a_ptr,\
  const gemm##_skinnyger_bscalar *b_ptr,\
  gemm##_skinnyger_cscalar *c_ptr,\
  uint32_t m_left, uint32_t LDA, uint32_t LDB, bool b_rowmajor) {\
\
  MACRO_EXP_##k_vlen(VOID_BASE, GEMM_SKINNY_GER_INIT_APTR_ITEM, gemm)\
  MACRO_EXP_##n_dim(VOID_BASE, GEMM_SKINNY_GER_DECLARE_B_ITEM, gemm, k_vlen)\
  if (b_rowmajor) {\
    MACRO_EXP_##n_dim(VOID_BASE,\
      GEMM_SKINNY_GER_LOADB_BROWMAJOR_ITEM, gemm, k_vlen)\
  } else {\
    MACRO_EXP_##n_dim(VOID_BASE,\
      GEMM_SKINNY_GER_LOADB_BCOLMAJOR_ITEM, gemm, k_vlen)\
  }\
\
  MACRO_EXP_M_##m_mask(GEMM_SKINNY_GER_COMPUTE_BLOCK_LOOP,\
    gemm, k_vlen, n_dim)\
}

#define GEMM_SKINNY_GER_INLINE_FUNC_ITEM(k_vlen, gemm, n_dim, m_mask)\
  GEMM_SKINNY_GER_INLINE_FUNC(gemm, n_dim, k_vlen, m_mask)

#define GEMM_SKINNY_GER_INLINE_FUNCS(gemm, n_dim, k_mask, m_mask) \
  MACRO_EXPANSION_M_##k_mask(GEMM_SKINNY_GER_INLINE_FUNC_ITEM, gemm, n_dim, m_mask)

#define GEMM_SKINNY_GER_INLINE_CALL_LOOP(k_vlen, gemm, n_dim) \
  for (; k_left >= k_vlen; k_left -= k_vlen) {\
    inline_##gemm##_acolmajor_bskinny_k##k_vlen##n##n_dim(\
      a_ptr, b_ptr, c_scratch, m_inc, M, LDB, b_rowmajor);\
    a_ptr += k_vlen * M;\
    b_ptr += k_vlen * b_k_inc;\
  }

#define GEMM_SKINNY_GER_BETA_FUNC(gemm, n_dim) \
static inline void inline_##gemm##_acolmajor_bskinny_beta_##n_dim(\
  gemm##_skinnyger_cscalar *c_ptr, uint32_t M,\
  gemm##_skinnyger_cscalar beta) {\
\
  if (beta == (gemm##_skinnyger_cscalar)1.0) {\
    return;\
  }\
\
  uint64_t size = (uint64_t)M * n_dim;\
  for (; size > 7; size -= 8) {\
    c_ptr[0] *= beta; c_ptr[1] *= beta;\
    c_ptr[2] *= beta; c_ptr[3] *= beta;\
    c_ptr[4] *= beta; c_ptr[5] *= beta;\
    c_ptr[6] *= beta; c_ptr[7] *= beta;\
    c_ptr += 8;\
  }\
  for (; size > 0; size--) {\
    *c_ptr *= beta;\
    c_ptr++;\
  }\
}

/* params atype & btype here are for function name mangling only */
#define GEMM_SKINNY_GER_SERIAL_FUNC(gemm, n_dim,\
  k_mask, m_mask, stack_size, atype, btype) \
GEMM_SKINNY_GER_BETA_FUNC(gemm, n_dim)\
GEMM_SKINNY_GER_INLINE_FUNCS(gemm, n_dim, k_mask, m_mask)\
GEMM_SKINNY_GER_INLINE_DEPACK_FUNC(gemm, m_mask, n_dim)\
void gemm##_acolmajor_bskinny_a##atype##_b##btype##_n##n_dim(\
  const gemm##_skinnyger_ascalar *A,\
  const gemm##_skinnyger_bscalar *B,\
  gemm##_skinnyger_cscalar *C,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  gemm##_skinnyger_cscalar beta_inp) {\
\
  __attribute__((aligned(4096))) gemm##_skinnyger_cscalar\
    gemm##_acolmajor_bskinny_a##atype##_b##btype##_##n_dim##_cscratch[stack_size];\
\
  const bool b_rowmajor = b_c_order & 1;\
  const bool c_rowmajor = b_c_order & 2;\
  const uint32_t b_k_inc = b_rowmajor ? n_dim : 1;\
  const uint32_t LDB = b_rowmajor ? n_dim : K;\
\
  if (n_dim == 1) {\
    uint32_t k_left = K;\
    const uint32_t m_inc = M;\
    const gemm##_skinnyger_ascalar *a_ptr = A;\
    const gemm##_skinnyger_bscalar *b_ptr = B;\
    gemm##_skinnyger_cscalar *c_scratch = C;\
    inline_##gemm##_acolmajor_bskinny_beta_##n_dim(c_scratch, M, beta_inp);\
    MACRO_EXP_M_##k_mask(GEMM_SKINNY_GER_INLINE_CALL_LOOP, gemm, 1)\
    return;\
  }\
\
  const uint32_t m_limit = ((stack_size / n_dim) >> 5) << 5;\
  uint32_t m_pos, m_inc;\
  for (m_pos = 0; m_pos < M; m_pos += m_inc) {\
    m_inc = M - m_pos;\
    if (m_inc >= (m_limit << 1)) m_inc = m_limit;\
    else if (m_inc > m_limit) m_inc >>= 1;\
    uint32_t k_left = K;\
    const gemm##_skinnyger_ascalar *a_ptr = A + m_pos;\
    const gemm##_skinnyger_bscalar *b_ptr = B;\
    gemm##_skinnyger_cscalar *c_scratch =\
      gemm##_acolmajor_bskinny_a##atype##_b##btype##_##n_dim##_cscratch;\
    memset(c_scratch, 0, m_inc * n_dim * sizeof(gemm##_skinnyger_cscalar));\
    MACRO_EXP_M_##k_mask(GEMM_SKINNY_GER_INLINE_CALL_LOOP, gemm, n_dim)\
    inline_##gemm##_acolmajor_bskinny_depack_c_n##n_dim(c_rowmajor, C,\
      c_scratch, M, m_pos, m_inc, beta_inp);\
  }\
}

#ifdef EMLL_SERIAL_ONLY

#define GEMM_SKINNY_GER_PARALLEL_FUNC(gemm, n_dim,\
  k_mask, m_mask, stack_size, atype, btype) \
GEMM_SKINNY_GER_SERIAL_FUNC(gemm, n_dim, k_mask, m_mask, stack_size, atype, btype)\
void gemm##_acolmajor_bskinny_a##atype##_b##btype##_n##n_dim##_omp(\
  const gemm##_skinnyger_ascalar *A,\
  const gemm##_skinnyger_bscalar *B,\
  gemm##_skinnyger_cscalar *C,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  gemm##_skinnyger_cscalar beta_inp, uint32_t num_threads) {\
\
  gemm##_acolmajor_bskinny_a##atype##_b##btype##_n##n_dim(\
    A, B, C, M, K, b_c_order, beta_inp);\
}

#else

/* params atype & btype here are for function name mangling only */
#define GEMM_SKINNY_GER_PARALLEL_FUNC(gemm, n_dim,\
  k_mask, m_mask, stack_size, atype, btype) \
struct gemm##_skinnyger_a##atype##_b##btype##_n##n_dim##_info {\
  const gemm##_skinnyger_ascalar *m_A;\
  const gemm##_skinnyger_bscalar *m_B;\
  gemm##_skinnyger_cscalar *m_C;\
  uint32_t m_M;\
};\
GEMM_SKINNY_GER_SERIAL_FUNC(gemm, n_dim, k_mask, m_mask, stack_size, atype, btype)\
void gemm##_acolmajor_bskinny_a##atype##_b##btype##_n##n_dim##_omp(\
  const gemm##_skinnyger_ascalar *A,\
  const gemm##_skinnyger_bscalar *B,\
  gemm##_skinnyger_cscalar *C,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  gemm##_skinnyger_cscalar beta_inp, uint32_t num_threads) {\
\
  if (num_threads <= 1) {\
    gemm##_acolmajor_bskinny_a##atype##_b##btype##_n##n_dim(\
      A, B, C, M, K, b_c_order, beta_inp);\
    return;\
  }\
\
  inline_##gemm##_acolmajor_bskinny_beta_##n_dim(C, M, beta_inp);\
  const bool b_rowmajor = b_c_order & 1;\
  const bool c_rowmajor = b_c_order & 2;\
  const uint32_t b_k_inc = b_rowmajor ? n_dim : 1;\
  const uint32_t LDB = b_rowmajor ? n_dim : K;\
  const uint32_t m_limit = ((stack_size / n_dim) >> 5) << 5;\
  const uint32_t m_task_min = m_limit >= 256 ? 256 : m_limit;\
  const uint64_t m_k_task_min = (16ULL << 32) | (uint64_t)m_task_min;\
  const uint64_t m_k_pos_max = ((uint64_t)K << 32) | (uint64_t)M;\
  uint64_t task_end = 0;\
\
  struct gemm##_skinnyger_a##atype##_b##btype##_n##n_dim##_info task_info;\
  task_info.m_A = A;\
  task_info.m_B = B;\
  task_info.m_C = C;\
  task_info.m_M = M;\
\
  omp_set_num_threads(num_threads);\
  _Pragma("omp parallel")\
  {\
    __attribute__((aligned(4096))) gemm##_skinnyger_cscalar\
      gemm##_acolmajor_bskinny_a##atype##_b##btype##_##n_dim##_cscratch[stack_size];\
    const gemm##_skinnyger_ascalar * const A = task_info.m_A;\
    const gemm##_skinnyger_bscalar * const B = task_info.m_B;\
    gemm##_skinnyger_cscalar * const C = task_info.m_C;\
    const uint32_t M = task_info.m_M;\
    uint32_t m_start, k_start, m_end, k_end, m_start_old, m_inc_old;\
    m_start_old = M; m_inc_old = 0;\
    gemm##_skinnyger_cscalar * const c_scratch = \
      gemm##_acolmajor_bskinny_a##atype##_b##btype##_##n_dim##_cscratch;\
    while(get_mn_task(&task_end, &m_start, &k_start, &m_end, &k_end,\
      m_k_task_min, m_limit, 0, m_k_pos_max, num_threads)) {\
\
      uint32_t k_left = k_end - k_start;\
      const uint32_t m_inc = m_end - m_start;\
      const gemm##_skinnyger_ascalar *a_ptr = A + k_start * M + m_start;\
      const gemm##_skinnyger_bscalar *b_ptr = B + k_start * b_k_inc;\
      if (m_start != m_start_old) {\
        if (m_inc_old > 0) {\
          _Pragma("omp critical")\
          {\
            inline_##gemm##_acolmajor_bskinny_depack_c_n##n_dim(c_rowmajor, C,\
              c_scratch, M, m_start_old, m_inc_old, 1);\
          }\
        }\
        memset(c_scratch, 0, m_inc * n_dim * sizeof(gemm##_skinnyger_cscalar));\
      }\
      MACRO_EXP_M_##k_mask(GEMM_SKINNY_GER_INLINE_CALL_LOOP, gemm, n_dim)\
      m_start_old = m_start; m_inc_old = m_inc;\
    }\
    if (m_inc_old > 0) {\
      _Pragma("omp critical")\
      {\
        inline_##gemm##_acolmajor_bskinny_depack_c_n##n_dim(c_rowmajor, C,\
          c_scratch, M, m_start_old, m_inc_old, 1);\
      }\
    }\
  }\
}

#endif

#define GEMM_SKINNY_GER_DEPACK_CRM_LOW_ITEM(n_id, m_id, m_vlen, n_dim) \
  c_wt[(m_id - 1) * n_dim + n_id - 1] =\
    c_wt[(m_id - 1) * n_dim + n_id - 1] * beta +\
      c_rd[(m_id - 1) + (n_id - 1) * m_vlen];

#define GEMM_SKINNY_GER_DEPACK_CRM_MID_ITEM(m_id, m_vlen, n_dim) \
  MACRO_EXPANSION_##n_dim(VOID_BASE,\
    GEMM_SKINNY_GER_DEPACK_CRM_LOW_ITEM, m_id, m_vlen, n_dim)

#define GEMM_SKINNY_GER_DEPACK_CRM_BLOCK_LOOP(m_vlen, gemm, n_dim) \
  for (; m_left >= m_vlen; m_left -= m_vlen) {\
    MACRO_EXP_##m_vlen(VOID_BASE,\
      GEMM_SKINNY_GER_DEPACK_CRM_MID_ITEM, m_vlen, n_dim)\
    c_wt += m_vlen * n_dim;\
    c_rd += m_vlen * n_dim;\
  }

#define GEMM_SKINNY_GER_DEPACK_CCM_LOW_ITEM(m_id, n_id, m_vlen) \
  c_wt1[m_id - 1] = c_wt1[m_id - 1] * beta +\
      c_rd[(n_id - 1) * m_vlen + m_id - 1];

#define GEMM_SKINNY_GER_DEPACK_CCM_MID_ITEM(n_id, m_vlen) \
  MACRO_EXPANSION_##m_vlen(VOID_BASE,\
    GEMM_SKINNY_GER_DEPACK_CCM_LOW_ITEM, n_id, m_vlen)\
  c_wt1 += M;

#define GEMM_SKINNY_GER_DEPACK_CCM_BLOCK_LOOP(m_vlen, gemm, n_dim) \
  for (; m_left >= m_vlen; m_left -= m_vlen) {\
    gemm##_skinnyger_cscalar *c_wt1 = c_wt;\
    MACRO_EXP_##n_dim(VOID_BASE,\
      GEMM_SKINNY_GER_DEPACK_CCM_MID_ITEM, m_vlen)\
    c_wt += m_vlen;\
    c_rd += m_vlen * n_dim;\
  }

#define GEMM_SKINNY_GER_INLINE_DEPACK_FUNC(gemm, m_mask, n_dim) \
static void inline_##gemm##_acolmajor_bskinny_depack_c_n##n_dim(\
  bool c_rowmajor, gemm##_skinnyger_cscalar * __restrict__ C,\
  const gemm##_skinnyger_cscalar * __restrict__ c_scratch,\
  uint32_t M, uint32_t m_pos, uint32_t m_left,\
  gemm##_skinnyger_cscalar beta) {\
\
  const gemm##_skinnyger_cscalar *c_rd = c_scratch;\
  if (c_rowmajor) {\
    gemm##_skinnyger_cscalar *c_wt = C + m_pos * n_dim;\
    MACRO_EXP_M_##m_mask(GEMM_SKINNY_GER_DEPACK_CRM_BLOCK_LOOP, gemm, n_dim)\
  } else {\
    gemm##_skinnyger_cscalar *c_wt = C + m_pos;\
    MACRO_EXP_M_##m_mask(GEMM_SKINNY_GER_DEPACK_CCM_BLOCK_LOOP, gemm, n_dim)\
  }\
}

#endif
