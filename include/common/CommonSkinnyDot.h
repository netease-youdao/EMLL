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
* File:        CommonSkinnyDot.h
* Description: Common building blocks for regular * skinny or skinny * regular
*              matmul when the regular matrix is row-major in the former
*              case or column-major in the latter case. These 2 kinds of matmul
*              involving skinny matrices require a special efficient kernel
*              different from that in regular * regular matmul. Specifically,
*              The regular matrix is no longer reordered (packed) during
*              calculation. Elements from the regular matrix are accessed
*              sequentially and only once.The GEMM calculation is decomposed
*              into sequential DOT operations.The skinny source matrix is
*              accessed repeatedly in DOT operations so it is always packed
*              in a scratch array.
* Extension:   To support a new CPU architecture, the following tasks should
*              be done in addition to including this header:
*              (1) Use typedef to define <gemm>_skinnydot_[a/b/c]scalar and
*                  <gemm>_skinnydot_[a/b/c]vec[\d]. For example, when
*                  developing avx2 SGEMM regular*skinny kernels, the following
*                  lines should be added when the maximum vector length is 8
*                  in K dimension:
*                    // scalar types in main memory
*                    typedef float sgemm_skinnydot_ascalar;
*                    typedef float sgemm_skinnydot_bscalar;
*                    typedef float sgemm_skinnydot_cscalar;
*                    // (converted) vector types in registers
*                    typedef float sgemm_skinnydot_avec1;
*                    typedef __m128 sgemm_skinnydot_avec4;
*                    typedef __m256 sgemm_skinnydot_avec8;
*                    typedef float sgemm_skinnydot_bvec1;
*                    typedef __m128 sgemm_skinnydot_bvec4;
*                    typedef __m256 sgemm_skinnydot_bvec8;
*                    typedef float sgemm_skinnydot_cvec1;
*                    typedef __m128 sgemm_skinnydot_cvec4;
*                    typedef __m256 sgemm_skinnydot_cvec8;
*              (2) Implement inline functions for basic vector-vector
*                  multiply-add operations. Here are examples for
*                  inline functions of avx2 SGEMM with k_veclen = 8, 4 and 1.
*                  These functions multiplies each element in a_vec with the
*                  corresponding element in b_vec and add the result
*                  to the corresponding element in c_vec:
*                    GEMM_SKINNY_DOT_CALC_UNIT(sgemm, 8) {
*                      return _mm256_fmadd_ps(a_vec, b_vec, c_vec);
*                    }
*                    GEMM_SKINNY_DOT_CALC_UNIT(sgemm, 4) {
*                      return _mm_fmadd_ps(a_vec, b_vec, c_vec);
*                    }
*                    GEMM_SKINNY_DOT_CALC_UNIT(sgemm, 1) {
*                      return a_vec * b_vec + c_vec;
*                    }
*              (3) Implement load and store inline functions for matrix
*                  a & b like this (each catagory 1 example (k_veclen = 8)):
*                    GEMM_SKINNY_DOT_LOADA_UNIT(sgemm, 8) {
*                      _mm_prefetch((char *)(a_ptr + 24), _MM_HINT_T0);
*                      return _mm256_loadu_ps(a_ptr);
*                    }
*                    GEMM_SKINNY_DOT_LOADB_UNIT(sgemm, 8) {
*                      return _mm256_loadu_ps(b_ptr);
*                    }
*              (4) Implement inline vectorized reduction functions:
*                    // reduction from vec[8] to vec[4]
*                    GEMM_SKINNY_DOT_REDUC_UNIT(sgemm, 8, 4) {
*                      return _mm_add_ps(_mm256_extractf128_ps(c_vec, 0),
*                        _mm256_extractf128_ps(c_vec, 1));
*                    }
*                    // reduction from vec[4] to vec[1]
*                    GEMM_SKINNY_DOT_REDUC_UNIT(sgemm, 4, 1) {
*                      __m128 z0 = _mm_setzero_ps();
*                      c_vec = _mm_hadd_ps(c_vec, z0);
*                      c_vec = _mm_hadd_ps(c_vec, z0);
*                      return _mm_cvtss_f32(c_vec);
*                    }
*              (5) Implement inline vector initialization functions.
*                  A function in this category returns a vector filled with
*                  zeros.
*                    GEMM_SKINNY_DOT_INITC_UNIT(sgemm, 8) {
*                      return _mm256_setzero_ps();
*                    }
*                    GEMM_SKINNY_DOT_INITC_UNIT(sgemm, 4) {
*                      return _mm_setzero_ps();
*                    }
*                    GEMM_SKINNY_DOT_INITC_UNIT(sgemm, 1) {
*                      return 0;
*                    }
*              (5) Finally build kernel functions from inline functions
*                  defined above. For each kernel function only 1 line
*                  is needed. The following line defines regular*skinny
*                  kernel functions (serial and OpenMP) for the minimum
*                  dimension length = 2 with k_veclen = {1, 4, 8} and
*                  m_unroll = {1, 2, 4}:
*                    GEMM_SKINNY_DOT_PARALLEL_FUNC(sgemm, 2, 13, 7, 8192,
*                      float, float)
*                  The last 2 parameters in the macro are for function
*                  name mangling, providing the data type for regular
*                  and skinny matrix respectively. The last number in
*                  macro parameters (8192) specify the scratch size
*                  for skinny matrix which should be adjusted to the size
*                  of L1 cache. The second number (13) is the sum of all
*                  implemented k_veclen values. The third number (7) is
*                  the sum of all m_unroll values covered.
******************************************************************************/

#include "common/ExpandMacro.h"
#include "common/CommonSched.h"

#include <stdbool.h>
#ifndef EMLL_SERIAL_ONLY
#include <omp.h>
#endif

#ifndef INCLUDE_COMMON_SKINNY_DOT
#define INCLUDE_COMMON_SKINNY_DOT

/* computation units basic in skinny_dot function */
#define GEMM_SKINNY_DOT_CALC_UNIT(gemm, k_veclen) \
static inline gemm##_skinnydot_cvec##k_veclen\
  inline_##gemm##_arowmajor_bskinny_fma_unit_m1n1k##k_veclen(\
    gemm##_skinnydot_cvec##k_veclen c_vec,\
    gemm##_skinnydot_avec##k_veclen a_vec,\
    gemm##_skinnydot_bvec##k_veclen b_vec)
/* you should give vectorized implementation equivalent to this:
 * GEMM_SKINNY_DOT_CALC_UNIT(gemm, k_veclen) {
 *   gemm##_skinnydot_cvec##k_veclen ret;
 *   for (int i = 0; i < k_veclen; ++i) {
 *     ret[i] = a_vec[i] * b_vec[i] + c_vec[i];
 *   }
 *   return ret;
 * }
 */

#define GEMM_SKINNY_DOT_LOADA_UNIT(gemm, k_veclen) \
static inline gemm##_skinnydot_avec##k_veclen\
  inline_##gemm##_arowmajor_bskinny_loada_unit_k##k_veclen(\
    const gemm##_skinnydot_ascalar *a_ptr)
/* you should give vectorized implementation equivalent to this:
 * GEMM_SKINNY_DOT_LOADA_UNIT(gemm, k_veclen) {
 *   gemm##_skinnydot_avec##k_veclen ret;
 *   for (int i = 0; i < k_veclen; ++i) {
 *     ret[i] = a_ptr[i];
 *   }
 *   prefetch(a_ptr + pref_distance);
 *   return ret;
 * }
 */

#define GEMM_SKINNY_DOT_LOADB_UNIT(gemm, k_veclen) \
static inline gemm##_skinnydot_bvec##k_veclen\
  inline_##gemm##_arowmajor_bskinny_loadb_unit_k##k_veclen(\
    const gemm##_skinnydot_bscalar *b_ptr)
/* you should give vectorized implementation equivalent to this:
 * GEMM_SKINNY_DOT_LOADB_UNIT(gemm, k_veclen) {
 *   gemm##_skinnydot_bvec##k_veclen ret;
 *   for (int i = 0; i < k_veclen; ++i) {
 *     ret[i] = b_ptr[i];
 *   }
 *   return ret;
 * }
 */

#define GEMM_SKINNY_DOT_REDUC_UNIT(gemm, old_k_vlen, new_k_vlen) \
static inline gemm##_skinnydot_cvec##new_k_vlen\
  inline_##gemm##_arowmajor_bskinny_reduc_unit_##new_k_vlen##from##old_k_vlen(\
    gemm##_skinnydot_cvec##old_k_vlen c_vec)
/* The sum of all elements of the returned vector should be
 * equal to that of the input c_vec, here's an example:
 * GEMM_SKINNY_DOT_REDUC_UNIT(gemm, old_k_vlen, new_k_vlen) {
 *   gemm##_skinnydot_cvec##new_k_vlen ret;
 *   int i;
 *   for (i = 0; i < new_k_vlen; ++i) {
 *     ret[i] = c_vec[i];
 *   }
 *   for (; i < old_k_vlen; ++i) {
 *     ret[i % new_k_vlen] += c_vec[i];
 *   }
 *   return ret;
 * }
 */

#define GEMM_SKINNY_DOT_INITC_UNIT(gemm, k_veclen) \
static inline gemm##_skinnydot_cvec##k_veclen\
  inline_##gemm##_arowmajor_bskinny_initc_unit_k##k_veclen()
/* you should give vectorized implementation equivalent to this:
 * GEMM_SKINNY_DOT_INITC_UNIT(gemm, k_veclen) {
 *   gemm##_skinnydot_cvec##k_veclen ret = {0};
 *   return ret;
 * }
 */

/* construct inline function from building blocks */
#define GEMM_SKINNY_DOT_INIT_CVEC_ITEM(m_id, gemm, k_veclen, n_id) \
  gemm##_skinnydot_cvec##k_veclen c_##k_veclen##_##m_id##_##n_id =\
    inline_##gemm##_arowmajor_bskinny_initc_unit_k##k_veclen();

#define GEMM_SKINNY_DOT_INIT_CVEC_COL_ITEM(n_id, gemm, k_veclen, m_unroll) \
  MACRO_EXPANSION_##m_unroll(VOID_BASE, GEMM_SKINNY_DOT_INIT_CVEC_ITEM,\
    gemm, k_veclen, n_id)

#define GEMM_SKINNY_DOT_INIT_CVEC(k_veclen, gemm, m_unroll, n_dim) \
  MACRO_EXP_##n_dim(VOID_BASE, GEMM_SKINNY_DOT_INIT_CVEC_COL_ITEM,\
    gemm, k_veclen, m_unroll)

#define GEMM_SKINNY_DOT_CALC_ITEM(m_id, gemm, k_veclen, n_id) \
  c_##k_veclen##_##m_id##_##n_id =\
    inline_##gemm##_arowmajor_bskinny_fma_unit_m1n1k##k_veclen(\
      c_##k_veclen##_##m_id##_##n_id, a_##k_veclen##_##m_id,\
      b_##k_veclen##_##n_id);

#define GEMM_SKINNY_DOT_CALC_COL_ITEM_PACK(n_id, gemm, k_veclen, m_unroll) \
  const gemm##_skinnydot_bvec##k_veclen b_##k_veclen##_##n_id =\
    inline_##gemm##_arowmajor_bskinny_loadb_unit_k##k_veclen(\
      b_ptr + (n_id - 1) * k_veclen);\
  MACRO_EXPANSION_##m_unroll(VOID_BASE, GEMM_SKINNY_DOT_CALC_ITEM,\
    gemm, k_veclen, n_id)

#define GEMM_SKINNY_DOT_LOADA_ITEM(m_id, gemm, k_veclen) \
  const gemm##_skinnydot_avec##k_veclen a_##k_veclen##_##m_id =\
    inline_##gemm##_arowmajor_bskinny_loada_unit_k##k_veclen(a_ptr##m_id);\
  a_ptr##m_id += k_veclen;


#define GEMM_SKINNY_DOT_CALC_LOOPITEM_PACK(k_veclen, gemm, m_unroll, n_dim) \
for (; k_left >= k_veclen; k_left -= k_veclen) {\
  MACRO_EXP_##m_unroll(VOID_BASE, GEMM_SKINNY_DOT_LOADA_ITEM, gemm, k_veclen)\
  MACRO_EXP_##n_dim(VOID_BASE, GEMM_SKINNY_DOT_CALC_COL_ITEM_PACK,\
    gemm, k_veclen, m_unroll)\
  b_ptr += n_dim * k_veclen;\
}

#define GEMM_SKINNY_DOT_REDUC_ITEM(m_id, old_kvlen, new_kvlen, gemm, n_id) \
  gemm##_skinnydot_cvec##new_kvlen c_##new_kvlen##_##m_id##_##n_id =\
    inline_##gemm##_arowmajor_bskinny_reduc_unit_##new_kvlen##from##old_kvlen(\
      c_##old_kvlen##_##m_id##_##n_id);

#define GEMM_SKINNY_DOT_REDUC_COL_ITEM(n_id, m_unroll, gemm,\
  old_kvlen, new_kvlen) \
  MACRO_EXPANSION_##m_unroll(VOID_BASE, GEMM_SKINNY_DOT_REDUC_ITEM,\
    old_kvlen, new_kvlen, gemm, n_id)

#define GEMM_SKINNY_DOT_REDUC_CROSS_ITEM(old_kvlen, new_kvlen, gemm,\
  m_unroll, n_dim)\
  MACRO_EXP_##n_dim(VOID_BASE, GEMM_SKINNY_DOT_REDUC_COL_ITEM,\
    m_unroll, gemm, old_kvlen, new_kvlen)

#define GEMM_SKINNY_DOT_INIT_APTR_ITEM(m_id, gemm) \
  const gemm##_skinnydot_ascalar *a_ptr##m_id = A + (m_id - 1) * LDK;

#define GEMM_SKINNY_DOT_STOREC_ITEM_CC(m_id, n_id) \
  c_ptr[m_id - 1] = c_ptr[m_id - 1] * beta + c_1_##m_id##_##n_id;

#define GEMM_SKINNY_DOT_STOREC_ITEM_CR(n_id, m_id) \
  c_ptr[n_id - 1] = c_ptr[n_id - 1] * beta + c_1_##m_id##_##n_id;

#define GEMM_SKINNY_DOT_STOREC_CC_COL_ITEM(n_id, m_unroll) \
  MACRO_EXPANSION_##m_unroll(VOID_BASE, GEMM_SKINNY_DOT_STOREC_ITEM_CC, n_id)\
  c_ptr += LDM;

#define GEMM_SKINNY_DOT_STOREC_CR_ROW_ITEM(m_id, n_dim) \
  MACRO_EXPANSION_##n_dim(VOID_BASE, GEMM_SKINNY_DOT_STOREC_ITEM_CR, m_id)\
  c_ptr += n_dim;

#define GEMM_SKINNY_DOT_INLINE_PACK_FUNC(gemm, m_unroll, n_dim, k_mask) \
static inline void\
  inline_##gemm##_arowmajor_bskinny_m##m_unroll##n##n_dim(\
    const gemm##_skinnydot_ascalar *A, const gemm##_skinnydot_bscalar *b_ptr,\
    gemm##_skinnydot_cscalar *c_ptr, uint32_t k_left, uint32_t LDK, uint32_t LDM,\
    gemm##_skinnydot_cscalar beta, bool c_rowmajor) {\
\
  MACRO_EXP_##m_unroll(VOID_BASE, GEMM_SKINNY_DOT_INIT_APTR_ITEM, gemm)\
  MACRO_EXPANSION_IMX_##k_mask(GEMM_SKINNY_DOT_INIT_CVEC,\
    GEMM_SKINNY_DOT_CALC_LOOPITEM_PACK,\
    GEMM_SKINNY_DOT_REDUC_CROSS_ITEM, gemm, m_unroll, n_dim)\
  if (c_rowmajor) {\
    MACRO_EXP_##m_unroll(VOID_BASE, GEMM_SKINNY_DOT_STOREC_CR_ROW_ITEM, n_dim)\
  } else {\
    MACRO_EXP_##n_dim(VOID_BASE, GEMM_SKINNY_DOT_STOREC_CC_COL_ITEM, m_unroll)\
  }\
}

#define GEMM_SKINNY_DOT_INLINE_FUNC_ITEM(m_unroll, gemm, n_dim, k_mask) \
  GEMM_SKINNY_DOT_INLINE_PACK_FUNC(gemm, m_unroll, n_dim, k_mask)

#define GEMM_SKINNY_DOT_PACKK_BC_ITEM(k_id, n_id) \
  sb_ptr[k_id - 1] = b_ptr##n_id[k_id - 1];

#define GEMM_SKINNY_DOT_PACKK_BC_COL_ITEM(n_id, k_veclen) \
  MACRO_EXPANSION_##k_veclen(VOID_BASE, GEMM_SKINNY_DOT_PACKK_BC_ITEM, n_id)\
  b_ptr##n_id += k_veclen; sb_ptr += k_veclen;

#define GEMM_SKINNY_DOT_PACKK_BC_LOOP(k_veclen, n_dim) \
  for (; k_left >= k_veclen; k_left -= k_veclen) {\
    MACRO_EXP_##n_dim(VOID_BASE, GEMM_SKINNY_DOT_PACKK_BC_COL_ITEM, k_veclen)\
  }

#define GEMM_SKINNY_DOT_PACKK_BR_ITEM(n_id, k_id, k_veclen) \
  sb_ptr[(n_id - 1) * k_veclen + k_id - 1] = b_ptr[n_id - 1];

#define GEMM_SKINNY_DOT_PACKK_BR_ROW_ITEM(k_id, n_dim, k_veclen) \
  MACRO_EXPANSION_##n_dim(VOID_BASE, GEMM_SKINNY_DOT_PACKK_BR_ITEM,\
    k_id, k_veclen)\
  b_ptr += n_dim;

#define GEMM_SKINNY_DOT_PACKK_BR_LOOP(k_veclen, n_dim) \
  for (; k_left >= k_veclen; k_left -= k_veclen) {\
    MACRO_EXP_##k_veclen(VOID_BASE, GEMM_SKINNY_DOT_PACKK_BR_ROW_ITEM,\
      n_dim, k_veclen)\
    sb_ptr += n_dim * k_veclen;\
  }

#define GEMM_SKINNY_DOT_PACKK_BC_INIT_BPTR_ITEM(n_id, gemm) \
  const gemm##_skinnydot_bscalar *b_ptr##n_id = b_ptr + (n_id - 1) * K;

#define GEMM_SKINNY_DOT_INLINE_CALL_LOOP(m_unroll, gemm, n_dim) \
  if (unroll_m##m_unroll) {\
    for (; m_left >= m_unroll; m_left -= m_unroll) {\
      inline_##gemm##_arowmajor_bskinny_m##m_unroll##n##n_dim(\
        a_ptr, b_ptr, c_ptr, k_inc, K, M, beta, c_rowmajor);\
      a_ptr += K * m_unroll;\
      c_ptr += c_m_inc * m_unroll;\
    }\
  }

#define GEMM_SKINNY_DOT_UNROLL_TEST(m_unroll, unroll_test, n_dim) \
  const bool unroll_m##m_unroll = unroll_test##_m##m_unroll##n##n_dim(M, K)\
    || (m_unroll == 1);

#define GEMM_SKINNY_DOT_UNROLL_TEST_DEFAULT(m_unroll, n_dim) \
static inline bool unroll_test_m##m_unroll##n##n_dim(uint32_t M, uint32_t K) {\
  return true;\
}

#define GEMM_SKINNY_DOT_SERIAL_FUNC_NOINCINLINE(gemm, n_dim, k_mask, m_mask,\
  scratch_size, atype, btype, unroll_test) \
__attribute__((aligned(4096))) static __thread gemm##_skinnydot_bscalar\
  blas_skinny_dot_b_scratch_##btype##n##n_dim[scratch_size];\
void gemm##_arowmajor_bskinny_a##atype##_b##btype##_n##n_dim(\
  const gemm##_skinnydot_ascalar *A, const gemm##_skinnydot_bscalar *B,\
  gemm##_skinnydot_cscalar *C, uint32_t M, uint32_t K,\
  uint8_t b_c_order, gemm##_skinnydot_cscalar beta_inp) {\
\
  if (K == 0) {\
    if (beta_inp != (gemm##_skinnydot_cscalar)1.0) {\
      uint64_t size = (uint64_t)M * n_dim;\
      for (uint64_t pos = 0; pos < size; ++pos) {\
        C[pos] *= beta_inp;\
      }\
    }\
    return;\
  }\
\
  const bool b_rowmajor = b_c_order & 1;\
  const bool c_rowmajor = b_c_order & 2;\
  const uint32_t k_limit = ((scratch_size / n_dim) >> 5) << 5;\
  const uint32_t c_m_inc = c_rowmajor ? n_dim : 1;\
  MACRO_EXPANSION_M_##m_mask(GEMM_SKINNY_DOT_UNROLL_TEST, unroll_test, n_dim)\
\
  uint32_t k_pos, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= (k_limit << 1)) k_inc = k_limit;\
    else if (k_inc > k_limit) k_inc >>= 1;\
\
    const gemm##_skinnydot_cscalar beta = (k_pos == 0) ? beta_inp : 1;\
    if (n_dim == 1) {\
      const gemm##_skinnydot_ascalar *a_ptr = A + k_pos;\
      const gemm##_skinnydot_bscalar * const b_ptr = B + k_pos;\
      gemm##_skinnydot_cscalar *c_ptr = C;\
      uint32_t m_left = M;\
      MACRO_EXPANSION_M_##m_mask(GEMM_SKINNY_DOT_INLINE_CALL_LOOP, gemm, n_dim)\
    } else {\
      if (b_rowmajor) {\
        const gemm##_skinnydot_bscalar *b_ptr = B + k_pos * n_dim;\
        gemm##_skinnydot_bscalar *sb_ptr =\
          blas_skinny_dot_b_scratch_##btype##n##n_dim;\
        uint32_t k_left = k_inc;\
        MACRO_EXPANSION_M_##k_mask(GEMM_SKINNY_DOT_PACKK_BR_LOOP, n_dim)\
      } else {\
        const gemm##_skinnydot_bscalar *b_ptr = B + k_pos;\
        MACRO_EXP_##n_dim(VOID_BASE, GEMM_SKINNY_DOT_PACKK_BC_INIT_BPTR_ITEM, gemm)\
        gemm##_skinnydot_bscalar *sb_ptr =\
          blas_skinny_dot_b_scratch_##btype##n##n_dim;\
        uint32_t k_left = k_inc;\
        MACRO_EXPANSION_M_##k_mask(GEMM_SKINNY_DOT_PACKK_BC_LOOP, n_dim)\
      }\
      const gemm##_skinnydot_ascalar *a_ptr = A + k_pos;\
      const gemm##_skinnydot_bscalar * const b_ptr =\
        blas_skinny_dot_b_scratch_##btype##n##n_dim;\
      gemm##_skinnydot_cscalar *c_ptr = C;\
      uint32_t m_left = M;\
      MACRO_EXPANSION_M_##m_mask(GEMM_SKINNY_DOT_INLINE_CALL_LOOP, gemm, n_dim)\
    }\
  }\
}

/******************************************************************************
 * Template:    GEMM_SKINNY_DOT_SERIAL_FUNC
 * Description: Construct serial dot-based "regular * skinny" GEMM function
 *              from the general algorithm.
 * Parameters: gemm: The type of GEMM, e.g. sgemm, hgemm, u8u32gemm, ...
 *             n_dim: The width of skinny matrix that this function can handle.
 *                    (Every such function can only process 1 width)
 *             k_mask: The sum of all supported accumulation vector width.
 *                     For example, if inline calculation functions with
 *                     k_veclen = 1, 4 and 8 are available, this parameter
 *                     should be 1 + 4 + 8 = 13. Note that every k_veclen
 *                     should be a power of 2.
 *             m_mask: The sum of all supported unroll factors of M. During the
 *                     calculation of dot values, usually several rows are
 *                     read concurrently from the regular matrix to improve
 *                     the ratio of arith/load. But if too many rows are loaded
 *                     at the same time, there will be no enough registers to
 *                     hold dot values. So there's a balance. Let's say, if
 *                     the optimal solution is to read 4 rows together in the
 *                     bulk region and read one by one at the edge, this
 *                     parameter can be set to 4 + 1 = 5.
 *             scratch_size: The size (number of elements) for the scratch
 *                           array that holds rearranged (packed) block from
 *                           the skinny source matrix. Because the skinny
 *                           source is accessed repeatedly during calculations,
 *                           it's better to rearrange it to make the access
 *                           to its element fully sequential. This parameter
 *                           should not exceed the capacity of level-2 cache.
 *             atype: The data type of regular source matrix. This parameter
 *                    is only for naming the function properly so that it
 *                    can be called correctly by driver.
 *             btype: The data type of skinny source matrix. This parameter
 *                    is for naming the function only.
 *****************************************************************************/
#define GEMM_SKINNY_DOT_SERIAL_FUNC(gemm, n_dim, k_mask, m_mask,\
  scratch_size, atype, btype) \
  MACRO_EXP_M_##m_mask(GEMM_SKINNY_DOT_INLINE_FUNC_ITEM, gemm, n_dim, k_mask)\
  MACRO_EXP_M_##m_mask(GEMM_SKINNY_DOT_UNROLL_TEST_DEFAULT, n_dim)\
  GEMM_SKINNY_DOT_SERIAL_FUNC_NOINCINLINE(gemm, n_dim, k_mask, m_mask,\
    scratch_size, atype, btype, unroll_test)

#ifdef EMLL_SERIAL_ONLY

#define GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(gemm, n_dim, k_mask, m_mask,\
  scratch_size, atype, btype, unroll_test) \
GEMM_SKINNY_DOT_SERIAL_FUNC_NOINCINLINE(gemm, n_dim, k_mask, m_mask,\
  scratch_size, atype, btype, unroll_test) \
void gemm##_arowmajor_bskinny_a##atype##_b##btype##_n##n_dim##_omp(\
  const gemm##_skinnydot_ascalar *A, const gemm##_skinnydot_bscalar *B,\
  gemm##_skinnydot_cscalar *C, uint32_t M, uint32_t K,\
  uint8_t b_c_order, gemm##_skinnydot_cscalar beta_inp, uint32_t num_threads) {\
\
  gemm##_arowmajor_bskinny_a##atype##_b##btype##_n##n_dim(A, B, C,\
    M, K, b_c_order, beta_inp);\
}

#else

#define GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(gemm, n_dim, k_mask, m_mask,\
  scratch_size, atype, btype, unroll_test) \
GEMM_SKINNY_DOT_SERIAL_FUNC_NOINCINLINE(gemm, n_dim, k_mask, m_mask,\
  scratch_size, atype, btype, unroll_test) \
/* in ARMv7, the arguments when creating a thread is limited to a certain */\
/* number, so some arguments need to be wrapped into a struct to pass */\
struct gemm##_skinnydot_a##atype##_b##btype##_##n_dim##_matrix_info {\
  const gemm##_skinnydot_ascalar *m_A;\
  const gemm##_skinnydot_bscalar *m_B;\
  gemm##_skinnydot_cscalar *m_C;\
  uint32_t m_M;\
};\
void gemm##_arowmajor_bskinny_a##atype##_b##btype##_n##n_dim##_omp(\
  const gemm##_skinnydot_ascalar *A, const gemm##_skinnydot_bscalar *B,\
  gemm##_skinnydot_cscalar *C, uint32_t M, uint32_t K,\
  uint8_t b_c_order, gemm##_skinnydot_cscalar beta_inp, uint32_t num_threads) {\
\
  if (num_threads <= 1 || K == 0) {\
    gemm##_arowmajor_bskinny_a##atype##_b##btype##_n##n_dim(A, B, C,\
      M, K, b_c_order, beta_inp);\
    return;\
  }\
\
  struct gemm##_skinnydot_a##atype##_b##btype##_##n_dim##_matrix_info thread_args;\
  thread_args.m_A = A;\
  thread_args.m_B = B;\
  thread_args.m_C = C;\
  thread_args.m_M = M;\
  /* use the tls scratch of master thread for shared buffer */\
  gemm##_skinnydot_bscalar * const b_scratch_master =\
    blas_skinny_dot_b_scratch_##btype##n##n_dim;\
  const bool b_rowmajor = b_c_order & 1;\
  const bool c_rowmajor = b_c_order & 2;\
  const uint32_t k_limit = ((scratch_size / n_dim) >> 5) << 5;\
  const uint32_t c_m_inc = c_rowmajor ? n_dim : 1;\
  MACRO_EXPANSION_M_##m_mask(GEMM_SKINNY_DOT_UNROLL_TEST, unroll_test, n_dim)\
\
  uint32_t k_pos, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= (k_limit << 1)) k_inc = k_limit;\
    else if (k_inc > k_limit) k_inc >>= 1;\
\
    const gemm##_skinnydot_cscalar beta = (k_pos == 0) ? beta_inp : 1;\
    if (n_dim == 1) {\
      uint32_t m_done = 0;\
      omp_set_num_threads(num_threads);\
      _Pragma("omp parallel")\
      {\
        uint32_t m_start, m_end;\
        while(get_irreg_task(&m_done, &m_start, &m_end,\
          ((((M - m_done) / num_threads) >> 2) / MACRO_EXP_M_FIRSTITEM_##m_mask + 1)\
            * MACRO_EXP_M_FIRSTITEM_##m_mask, M)) {\
          const gemm##_skinnydot_ascalar *a_ptr = A + k_pos + m_start * K;\
          const gemm##_skinnydot_bscalar * const b_ptr = B + k_pos;\
          gemm##_skinnydot_cscalar *c_ptr = C + m_start;\
          uint32_t m_left = m_end - m_start;\
          MACRO_EXPANSION_M_##m_mask(GEMM_SKINNY_DOT_INLINE_CALL_LOOP, gemm, n_dim)\
        }\
      }\
    } else {\
      uint32_t m_done = 0;\
      uint32_t k_left_shared = k_inc;\
      omp_set_num_threads(num_threads);\
      _Pragma("omp parallel")\
      {\
        const gemm##_skinnydot_ascalar * const A = thread_args.m_A;\
        const gemm##_skinnydot_bscalar * const B = thread_args.m_B;\
        gemm##_skinnydot_cscalar * const C = thread_args.m_C;\
        const uint32_t M = thread_args.m_M;\
        uint32_t k_start, k_end;\
        while(get_copy_task(&k_left_shared, MACRO_EXP_M_FIRSTITEM_##k_mask << 3,\
          &k_start, &k_end)) {\
          if (b_rowmajor) {\
            const gemm##_skinnydot_bscalar *b_ptr = B + (k_pos + k_start) * n_dim;\
            gemm##_skinnydot_bscalar *sb_ptr = b_scratch_master + k_start * n_dim;\
            uint32_t k_left = k_end - k_start;\
            MACRO_EXPANSION_M_##k_mask(GEMM_SKINNY_DOT_PACKK_BR_LOOP, n_dim)\
          } else {\
            const gemm##_skinnydot_bscalar *b_ptr = B + k_pos + k_start;\
            MACRO_EXP_##n_dim(VOID_BASE, GEMM_SKINNY_DOT_PACKK_BC_INIT_BPTR_ITEM, gemm)\
            gemm##_skinnydot_bscalar *sb_ptr = b_scratch_master + k_start * n_dim;\
            uint32_t k_left = k_end - k_start;\
            MACRO_EXPANSION_M_##k_mask(GEMM_SKINNY_DOT_PACKK_BC_LOOP, n_dim)\
          }\
        }\
        _Pragma("omp barrier")\
        uint32_t m_start, m_end;\
        while(get_irreg_task(&m_done, &m_start, &m_end,\
          ((((M - m_done) / num_threads) >> 2) / MACRO_EXP_M_FIRSTITEM_##m_mask + 1)\
            * MACRO_EXP_M_FIRSTITEM_##m_mask, M)) {\
          const gemm##_skinnydot_ascalar *a_ptr = A + k_pos + m_start * K;\
          const gemm##_skinnydot_bscalar * const b_ptr = b_scratch_master;\
          gemm##_skinnydot_cscalar *c_ptr = C + c_m_inc * m_start;\
          uint32_t m_left = m_end - m_start;\
          MACRO_EXPANSION_M_##m_mask(GEMM_SKINNY_DOT_INLINE_CALL_LOOP,\
            gemm, n_dim)\
        }\
      }\
    }\
  }\
}

#endif

/******************************************************************************
 * Template:    GEMM_SKINNY_DOT_PARALLEL_FUNC
 * Description: Construct dot-based "regular * skinny" GEMM function
 *              paralleled by OpenMP.
 * Parameters: the same as in GEMM_SKINNY_DOT_SERIAL_FUNC
 *****************************************************************************/
#define GEMM_SKINNY_DOT_PARALLEL_FUNC(gemm, n_dim, k_mask, m_mask,\
  scratch_size, atype, btype) \
  MACRO_EXP_M_##m_mask(GEMM_SKINNY_DOT_INLINE_FUNC_ITEM, gemm, n_dim, k_mask)\
  MACRO_EXP_M_##m_mask(GEMM_SKINNY_DOT_UNROLL_TEST_DEFAULT, n_dim)\
  GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(gemm, n_dim, k_mask, m_mask,\
    scratch_size, atype, btype, unroll_test)

#endif
