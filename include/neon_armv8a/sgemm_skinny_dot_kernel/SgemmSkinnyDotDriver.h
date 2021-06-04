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


#include "arm_neon/ARMCompareAndSwap.h"
#include "common/CommonSched.h"
#ifndef EMLL_SERIAL_ONLY
#include <omp.h>
#endif

#ifndef INCLUDE_SKINNY1_DRIVER
#define INCLUDE_SKINNY1_DRIVER

#define DRIVER_PURE_PACK_SERIAL(cpu, ndim, K_BATCH, pack_type, unroll_m) \
void sgemm_skinny1_arowmajor_n##ndim##_##cpu(const float * __restrict__ A,\
  const float * __restrict__ B, float * __restrict__ C,\
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,\
  uint8_t b_c_order, float beta_inp) {\
\
  const uint8_t b_rowmajor = b_c_order & 1;\
  const uint8_t c_rowmajor = b_c_order & 2;\
\
  __attribute__((aligned(4096))) float b_scr[ndim * K_BATCH];\
\
  uint32_t k_pos, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= K_BATCH * 2) k_inc = K_BATCH;\
    else if (k_inc > K_BATCH) k_inc >>= 1;\
    if (b_rowmajor == 0) {\
      pack_##pack_type##_from_cm(b_scr, B + k_pos, LDB, k_inc, ndim);\
    } else {\
      pack_##pack_type##_from_rm(b_scr, B + k_pos * LDB, LDB, k_inc, ndim);\
    }\
    uint32_t m_pos = M;\
    const float *a_ptr = A + k_pos;\
    float *c_ptr = C;\
    const uint32_t c_incr = (c_rowmajor == 0) ? 1 : LDC;\
    const float beta = (k_pos == 0) ? beta_inp : 1.0f;\
    for (; m_pos >= unroll_m; m_pos -= unroll_m) {\
      sgemm_skinny1_##cpu##_m##unroll_m##n##ndim(a_ptr, b_scr, c_ptr,\
        k_inc, LDA, LDC, c_rowmajor, &beta);\
      a_ptr += LDA * unroll_m;\
      c_ptr += c_incr * unroll_m;\
    }\
    for (; m_pos > 0; m_pos--) {\
      sgemm_skinny1_##cpu##_m1n##ndim(a_ptr, b_scr, c_ptr, k_inc, LDC,\
        c_rowmajor, beta);\
      a_ptr += LDA;\
      c_ptr += c_incr;\
    }\
  }\
}

#define DRIVER_PURE_PACK_OMP(cpu, ndim, K_BATCH, pack_type, unroll_m) \
void sgemm_skinny1_arowmajor_n##ndim##_##cpu##_omp(\
  const float * __restrict__ A,\
  const float * __restrict__ B, float * __restrict__ C,\
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,\
  uint8_t b_c_order, float beta_inp, uint32_t num_threads) {\
\
  if (num_threads <= 1) {\
    sgemm_skinny1_arowmajor_n##ndim##_##cpu(A, B, C, M, K,\
      LDA, LDB, LDC, b_c_order, beta_inp);\
    return;\
  }\
  omp_set_num_threads(num_threads);\
\
  const uint8_t b_rowmajor = b_c_order & 1;\
  const uint8_t c_rowmajor = b_c_order & 2;\
  const uint32_t c_m_inc = (c_rowmajor == 0) ? 1 : LDC;\
\
  __attribute__((aligned(4096))) float b_scr[ndim * K_BATCH];\
\
  uint32_t k_pos, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= K_BATCH * 2) k_inc = K_BATCH;\
    else if (k_inc > K_BATCH) k_inc >>= 1;\
    const float beta = (k_pos == 0) ? beta_inp : 1.0f;\
\
    uint32_t k_copy_left = k_inc;\
    uint32_t m_calc_done = 0;\
    _Pragma("omp parallel")\
    {\
      uint32_t k_copy_start, k_copy_end;\
      while(get_copy_task(&k_copy_left, 64, &k_copy_start, &k_copy_end)) {\
        if (b_rowmajor == 0) {\
          pack_##pack_type##_from_cm(b_scr + k_copy_start * ndim,\
            B + k_pos + k_copy_start, LDB,\
            k_copy_end - k_copy_start, ndim);\
        } else {\
          pack_##pack_type##_from_rm(b_scr + k_copy_start * ndim,\
            B + (k_pos + k_copy_start) * LDB, LDB,\
            k_copy_end - k_copy_start, ndim);\
        }\
      }\
      _Pragma("omp barrier")\
      uint32_t m_calc_start, m_calc_end;\
      while(get_irreg_task(&m_calc_done, &m_calc_start, &m_calc_end,\
        unroll_m << 2, M)) {\
        const float *a_ptr = A + m_calc_start * LDA + k_pos;\
        float *c_ptr = C + m_calc_start * c_m_inc;\
        uint32_t sub_m_left = m_calc_end - m_calc_start;\
        for (; sub_m_left >= unroll_m; sub_m_left -= unroll_m) {\
          sgemm_skinny1_##cpu##_m##unroll_m##n##ndim(a_ptr, b_scr, c_ptr,\
            k_inc, LDA, LDC, c_rowmajor, &beta);\
          a_ptr += LDA * unroll_m;\
          c_ptr += c_m_inc * unroll_m;\
        }\
        for (; sub_m_left > 0; sub_m_left--) {\
          sgemm_skinny1_##cpu##_m1n##ndim(a_ptr, b_scr, c_ptr, k_inc, LDC,\
            c_rowmajor, beta);\
          a_ptr += LDA;\
          c_ptr += c_m_inc;\
        }\
      }\
    }\
  }\
}

#define DRIVER_MIX2_PACK_SERIAL(cpu, ndim, K_BATCH, pack1, pack2, n_pack1, n_pack2, unroll_m) \
void sgemm_skinny1_arowmajor_n##ndim##_##cpu(const float * __restrict__ A,\
  const float * __restrict__ B, float * __restrict__ C,\
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,\
  uint8_t b_c_order, float beta_inp) {\
\
  const uint8_t b_rowmajor = b_c_order & 1;\
  const uint8_t c_rowmajor = b_c_order & 2;\
\
  __attribute__((aligned(4096))) float b_scr[ndim * K_BATCH];\
  float * const b_scr2 = b_scr + n_pack1 * K_BATCH;\
\
  uint32_t k_pos, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= K_BATCH * 2) k_inc = K_BATCH;\
    else if (k_inc > K_BATCH) k_inc >>= 1;\
    if (b_rowmajor == 0) {\
      pack_##pack1##_from_cm(b_scr, B + k_pos, LDB, k_inc, n_pack1);\
      pack_##pack2##_from_cm(b_scr2, B + k_pos + n_pack1 * LDB,\
        LDB, k_inc, n_pack2);\
    } else {\
      pack_##pack1##_from_rm(b_scr, B + k_pos * LDB, LDB, k_inc, n_pack1);\
      pack_##pack2##_from_rm(b_scr2, B + k_pos * LDB + n_pack1,\
        LDB, k_inc, n_pack2);\
    }\
    uint32_t m_pos = M;\
    const float *a_ptr = A + k_pos;\
    float *c_ptr1 = C;\
    float *c_ptr2 = (c_rowmajor == 0) ? C + n_pack1 * LDC : C + n_pack1;\
    const uint32_t c_incr = (c_rowmajor == 0) ? 1 : LDC;\
    const float beta = (k_pos == 0) ? beta_inp : 1.0f;\
    for (; m_pos >= unroll_m; m_pos -= unroll_m) {\
      sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack1(a_ptr, b_scr, c_ptr1,\
        k_inc, LDA, LDC, c_rowmajor, &beta);\
      sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack2(a_ptr, b_scr2, c_ptr2,\
        k_inc, LDA, LDC, c_rowmajor, &beta);\
      a_ptr += LDA * unroll_m;\
      c_ptr1 += c_incr * unroll_m;\
      c_ptr2 += c_incr * unroll_m;\
    }\
    for (; m_pos > 0; m_pos--) {\
      sgemm_skinny1_##cpu##_m1n##n_pack1(a_ptr, b_scr, c_ptr1, k_inc, LDC,\
        c_rowmajor, beta);\
      sgemm_skinny1_##cpu##_m1n##n_pack2(a_ptr, b_scr2, c_ptr2, k_inc, LDC,\
        c_rowmajor, beta);\
      a_ptr += LDA;\
      c_ptr1 += c_incr;\
      c_ptr2 += c_incr;\
    }\
  }\
}

#define DRIVER_MIX2_PACK_OMP(cpu, ndim, K_BATCH, pack1, pack2, n_pack1, n_pack2, unroll_m) \
void sgemm_skinny1_arowmajor_n##ndim##_##cpu##_omp(\
  const float * __restrict__ A,\
  const float * __restrict__ B, float * __restrict__ C,\
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,\
  uint8_t b_c_order, float beta_inp, uint32_t num_threads) {\
\
  if (num_threads <= 1) {\
    sgemm_skinny1_arowmajor_n##ndim##_##cpu(A, B, C, M, K,\
      LDA, LDB, LDC, b_c_order, beta_inp);\
    return;\
  }\
\
  const uint8_t b_rowmajor = b_c_order & 1;\
  const uint8_t c_rowmajor = b_c_order & 2;\
  const uint32_t c_m_inc = (c_rowmajor == 0) ? 1 : LDC;\
\
  __attribute__((aligned(4096))) float b_scr[ndim * K_BATCH];\
  float * const b_scr2 = b_scr + n_pack1 * K_BATCH;\
\
  uint32_t k_pos, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= K_BATCH * 2) k_inc = K_BATCH;\
    else if (k_inc > K_BATCH) k_inc >>= 1;\
    const float beta = (k_pos == 0) ? beta_inp : 1.0f;\
\
    uint32_t k_copy_left = k_inc;\
    uint32_t m_calc_done = 0;\
    _Pragma("omp parallel")\
    {\
      uint32_t k_copy_start, k_copy_end;\
      while(get_copy_task(&k_copy_left, 64, &k_copy_start, &k_copy_end)) {\
        if (b_rowmajor == 0) {\
          pack_##pack1##_from_cm(b_scr + k_copy_start * n_pack1,\
            B + (k_pos + k_copy_start), LDB,\
            k_copy_end - k_copy_start, n_pack1);\
          pack_##pack2##_from_cm(b_scr2 + k_copy_start * n_pack2,\
            B + (k_pos + k_copy_start) + n_pack1 * LDB, LDB,\
            k_copy_end - k_copy_start, n_pack2);\
        } else {\
          pack_##pack1##_from_rm(b_scr + k_copy_start * n_pack1,\
            B + (k_pos + k_copy_start) * LDB, LDB,\
            k_copy_end - k_copy_start, n_pack1);\
          pack_##pack2##_from_rm(b_scr2 + k_copy_start * n_pack2,\
            B + (k_pos + k_copy_start) * LDB + n_pack1, LDB,\
            k_copy_end - k_copy_start, n_pack2);\
        }\
      }\
      _Pragma("omp barrier")\
      uint32_t m_calc_start, m_calc_end;\
      while(get_irreg_task(&m_calc_done, &m_calc_start, &m_calc_end,\
        unroll_m << 2, M)) {\
        const float *a_ptr = A + m_calc_start * LDA + k_pos;\
        float *c_ptr1 = C + m_calc_start * c_m_inc;\
        float *c_ptr2 = (c_rowmajor == 0) ?\
          c_ptr1 + n_pack1 * LDC : c_ptr1 + n_pack1;\
        uint32_t sub_m_left = m_calc_end - m_calc_start;\
        for (; sub_m_left >= unroll_m; sub_m_left -= unroll_m) {\
          sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack1(a_ptr, b_scr, c_ptr1,\
            k_inc, LDA, LDC, c_rowmajor, &beta);\
          sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack2(a_ptr, b_scr2, c_ptr2,\
            k_inc, LDA, LDC, c_rowmajor, &beta);\
          a_ptr += LDA * unroll_m;\
          c_ptr1 += c_m_inc * unroll_m;\
          c_ptr2 += c_m_inc * unroll_m;\
        }\
        for (; sub_m_left > 0; sub_m_left--) {\
          sgemm_skinny1_##cpu##_m1n##n_pack1(a_ptr, b_scr, c_ptr1, k_inc, LDC,\
            c_rowmajor, beta);\
          sgemm_skinny1_##cpu##_m1n##n_pack2(a_ptr, b_scr2, c_ptr2, k_inc, LDC,\
            c_rowmajor, beta);\
          a_ptr += LDA;\
          c_ptr1 += c_m_inc;\
          c_ptr2 += c_m_inc;\
        }\
      }\
    }\
  }\
}

#define DRIVER_MIX3_PACK_SERIAL(cpu, ndim, K_BATCH, pack1, pack2, pack3, n_pack1, n_pack2, n_pack3, unroll_m) \
void sgemm_skinny1_arowmajor_n##ndim##_##cpu(const float * __restrict__ A,\
  const float * __restrict__ B, float * __restrict__ C,\
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,\
  uint8_t b_c_order, float beta_inp) {\
\
  const uint8_t b_rowmajor = b_c_order & 1;\
  const uint8_t c_rowmajor = b_c_order & 2;\
\
  __attribute__((aligned(4096))) float b_scr[ndim * K_BATCH];\
  float * const b_scr2 = b_scr + n_pack1 * K_BATCH;\
  float * const b_scr3 = b_scr2 + n_pack2 * K_BATCH;\
\
  uint32_t k_pos, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= K_BATCH * 2) k_inc = K_BATCH;\
    else if (k_inc > K_BATCH) k_inc >>= 1;\
    if (b_rowmajor == 0) {\
      pack_##pack1##_from_cm(b_scr, B + k_pos, LDB, k_inc, n_pack1);\
      pack_##pack2##_from_cm(b_scr2, B + k_pos + n_pack1 * LDB,\
        LDB, k_inc, n_pack2);\
      pack_##pack3##_from_cm(b_scr3, B + k_pos + (n_pack1 + n_pack2) * LDB,\
        LDB, k_inc, n_pack3);\
    } else {\
      pack_##pack1##_from_rm(b_scr, B + k_pos * LDB, LDB, k_inc, n_pack1);\
      pack_##pack2##_from_rm(b_scr2, B + k_pos * LDB + n_pack1,\
        LDB, k_inc, n_pack2);\
      pack_##pack3##_from_rm(b_scr3, B + k_pos * LDB + n_pack1 + n_pack2,\
        LDB, k_inc, n_pack3);\
    }\
    uint32_t m_pos = M;\
    const float *a_ptr = A + k_pos;\
    float *c_ptr1 = C;\
    float *c_ptr2 = (c_rowmajor == 0) ? C + n_pack1 * LDC : C + n_pack1;\
    float *c_ptr3 = (c_rowmajor == 0) ? C + (n_pack1 + n_pack2) * LDC :\
      C + n_pack1 + n_pack2;\
    const uint32_t c_incr = (c_rowmajor == 0) ? 1 : LDC;\
    const float beta = (k_pos == 0) ? beta_inp : 1.0f;\
    for (; m_pos >= unroll_m; m_pos -= unroll_m) {\
      sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack1(a_ptr, b_scr, c_ptr1,\
        k_inc, LDA, LDC, c_rowmajor, &beta);\
      sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack2(a_ptr, b_scr2, c_ptr2,\
        k_inc, LDA, LDC, c_rowmajor, &beta);\
      sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack3(a_ptr, b_scr3, c_ptr3,\
        k_inc, LDA, LDC, c_rowmajor, &beta);\
      a_ptr += LDA * unroll_m;\
      c_ptr1 += c_incr * unroll_m;\
      c_ptr2 += c_incr * unroll_m;\
      c_ptr3 += c_incr * unroll_m;\
    }\
    for (; m_pos > 0; m_pos--) {\
      sgemm_skinny1_##cpu##_m1n##n_pack1(a_ptr, b_scr, c_ptr1, k_inc, LDC,\
        c_rowmajor, beta);\
      sgemm_skinny1_##cpu##_m1n##n_pack2(a_ptr, b_scr2, c_ptr2, k_inc, LDC,\
        c_rowmajor, beta);\
      sgemm_skinny1_##cpu##_m1n##n_pack3(a_ptr, b_scr3, c_ptr3, k_inc, LDC,\
        c_rowmajor, beta);\
      a_ptr += LDA;\
      c_ptr1 += c_incr;\
      c_ptr2 += c_incr;\
      c_ptr3 += c_incr;\
    }\
  }\
}

#define DRIVER_MIX3_PACK_OMP(cpu, ndim, K_BATCH, pack1, pack2, pack3, n_pack1, n_pack2, n_pack3, unroll_m) \
void sgemm_skinny1_arowmajor_n##ndim##_##cpu##_omp(\
  const float * __restrict__ A,\
  const float * __restrict__ B, float * __restrict__ C,\
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,\
  uint8_t b_c_order, float beta_inp, uint32_t num_threads) {\
\
  if (num_threads <= 1) {\
    sgemm_skinny1_arowmajor_n##ndim##_##cpu(A, B, C, M, K,\
      LDA, LDB, LDC, b_c_order, beta_inp);\
    return;\
  }\
\
  const uint8_t b_rowmajor = b_c_order & 1;\
  const uint8_t c_rowmajor = b_c_order & 2;\
  const uint32_t c_m_inc = (c_rowmajor == 0) ? 1 : LDC;\
\
  __attribute__((aligned(4096))) float b_scr[ndim * K_BATCH];\
  float * const b_scr2 = b_scr + n_pack1 * K_BATCH;\
  float * const b_scr3 = b_scr2 + n_pack2 * K_BATCH;\
\
  uint32_t k_pos, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= K_BATCH * 2) k_inc = K_BATCH;\
    else if (k_inc > K_BATCH) k_inc >>= 1;\
    const float beta = (k_pos == 0) ? beta_inp : 1.0f;\
\
    uint32_t k_copy_left = k_inc;\
    uint32_t m_calc_done = 0;\
    _Pragma("omp parallel")\
    {\
      uint32_t k_copy_start, k_copy_end;\
      while(get_copy_task(&k_copy_left, 64, &k_copy_start, &k_copy_end)) {\
        if (b_rowmajor == 0) {\
          pack_##pack1##_from_cm(b_scr + k_copy_start * n_pack1,\
            B + (k_pos + k_copy_start), LDB,\
            k_copy_end - k_copy_start, n_pack1);\
          pack_##pack2##_from_cm(b_scr2 + k_copy_start * n_pack2,\
            B + (k_pos + k_copy_start) + n_pack1 * LDB, LDB,\
            k_copy_end - k_copy_start, n_pack2);\
          pack_##pack3##_from_cm(b_scr3 + k_copy_start * n_pack3,\
            B + (k_pos + k_copy_start) + (n_pack1 + n_pack2) * LDB, LDB,\
            k_copy_end - k_copy_start, n_pack3);\
        } else {\
          pack_##pack1##_from_rm(b_scr + k_copy_start * n_pack1,\
            B + (k_pos + k_copy_start) * LDB, LDB,\
            k_copy_end - k_copy_start, n_pack1);\
          pack_##pack2##_from_rm(b_scr2 + k_copy_start * n_pack2,\
            B + (k_pos + k_copy_start) * LDB + n_pack1, LDB,\
            k_copy_end - k_copy_start, n_pack2);\
          pack_##pack3##_from_rm(b_scr3 + k_copy_start * n_pack3,\
            B + (k_pos + k_copy_start) * LDB + n_pack1 + n_pack2, LDB,\
            k_copy_end - k_copy_start, n_pack3);\
        }\
      }\
      _Pragma("omp barrier")\
      uint32_t m_calc_start, m_calc_end;\
      while(get_irreg_task(&m_calc_done, &m_calc_start, &m_calc_end,\
        unroll_m << 2, M)) {\
        const float *a_ptr = A + m_calc_start * LDA + k_pos;\
        float *c_ptr1 = C + m_calc_start * c_m_inc;\
        float *c_ptr2 = (c_rowmajor == 0) ?\
          c_ptr1 + n_pack1 * LDC : c_ptr1 + n_pack1;\
        float *c_ptr3 = (c_rowmajor == 0) ?\
          c_ptr1 + (n_pack1 + n_pack2) * LDC : c_ptr1 + n_pack1 + n_pack2;\
        uint32_t sub_m_left = m_calc_end - m_calc_start;\
        for (; sub_m_left >= unroll_m; sub_m_left -= unroll_m) {\
          sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack1(a_ptr, b_scr, c_ptr1,\
            k_inc, LDA, LDC, c_rowmajor, &beta);\
          sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack2(a_ptr, b_scr2, c_ptr2,\
            k_inc, LDA, LDC, c_rowmajor, &beta);\
          sgemm_skinny1_##cpu##_m##unroll_m##n##n_pack3(a_ptr, b_scr3, c_ptr3,\
            k_inc, LDA, LDC, c_rowmajor, &beta);\
          a_ptr += LDA * unroll_m;\
          c_ptr1 += c_m_inc * unroll_m;\
          c_ptr2 += c_m_inc * unroll_m;\
          c_ptr3 += c_m_inc * unroll_m;\
        }\
        for (; sub_m_left > 0; sub_m_left--) {\
          sgemm_skinny1_##cpu##_m1n##n_pack1(a_ptr, b_scr, c_ptr1, k_inc, LDC,\
            c_rowmajor, beta);\
          sgemm_skinny1_##cpu##_m1n##n_pack2(a_ptr, b_scr2, c_ptr2, k_inc, LDC,\
            c_rowmajor, beta);\
          sgemm_skinny1_##cpu##_m1n##n_pack3(a_ptr, b_scr3, c_ptr3, k_inc, LDC,\
            c_rowmajor, beta);\
          a_ptr += LDA;\
          c_ptr1 += c_m_inc;\
          c_ptr2 += c_m_inc;\
          c_ptr3 += c_m_inc;\
        }\
      }\
    }\
  }\
}

#ifdef EMLL_SERIAL_ONLY

#define DRIVER_PURE_PACK(cpu, ndim, K_BATCH, pack_type, unroll_m) \
  DRIVER_PURE_PACK_SERIAL(cpu, ndim, K_BATCH, pack_type, unroll_m)\
void sgemm_skinny1_arowmajor_n##ndim##_##cpu##_omp(\
  const float * __restrict__ A,\
  const float * __restrict__ B, float * __restrict__ C,\
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,\
  uint8_t b_c_order, float beta_inp, uint32_t num_threads) {\
\
  sgemm_skinny1_arowmajor_n##ndim##_##cpu(A, B, C, M, K,\
    LDA, LDB, LDC, b_c_order, beta_inp);\
}

#define DRIVER_MIX2_PACK(cpu, ndim, K_BATCH, pack1, pack2, n_pack1, n_pack2, unroll_m) \
  DRIVER_MIX2_PACK_SERIAL(cpu, ndim, K_BATCH, pack1, pack2, n_pack1, n_pack2, unroll_m)\
void sgemm_skinny1_arowmajor_n##ndim##_##cpu##_omp(\
  const float * __restrict__ A,\
  const float * __restrict__ B, float * __restrict__ C,\
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,\
  uint8_t b_c_order, float beta_inp, uint32_t num_threads) {\
\
  sgemm_skinny1_arowmajor_n##ndim##_##cpu(A, B, C, M, K,\
    LDA, LDB, LDC, b_c_order, beta_inp);\
}

#define DRIVER_MIX3_PACK(cpu, ndim, K_BATCH, pack1, pack2, pack3, n_pack1, n_pack2, n_pack3, unroll_m) \
  DRIVER_MIX3_PACK_SERIAL(cpu, ndim, K_BATCH, pack1, pack2, pack3, n_pack1, n_pack2, n_pack3, unroll_m)\
void sgemm_skinny1_arowmajor_n##ndim##_##cpu##_omp(\
  const float * __restrict__ A,\
  const float * __restrict__ B, float * __restrict__ C,\
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,\
  uint8_t b_c_order, float beta_inp, uint32_t num_threads) {\
\
  sgemm_skinny1_arowmajor_n##ndim##_##cpu(A, B, C, M, K,\
    LDA, LDB, LDC, b_c_order, beta_inp);\
}

#else

#define DRIVER_PURE_PACK(cpu, ndim, K_BATCH, pack_type, unroll_m) \
  DRIVER_PURE_PACK_SERIAL(cpu, ndim, K_BATCH, pack_type, unroll_m)\
  DRIVER_PURE_PACK_OMP(cpu, ndim, K_BATCH, pack_type, unroll_m)

#define DRIVER_MIX2_PACK(cpu, ndim, K_BATCH, pack1, pack2, n_pack1, n_pack2, unroll_m) \
  DRIVER_MIX2_PACK_SERIAL(cpu, ndim, K_BATCH, pack1, pack2, n_pack1, n_pack2, unroll_m)\
  DRIVER_MIX2_PACK_OMP(cpu, ndim, K_BATCH, pack1, pack2, n_pack1, n_pack2, unroll_m)

#define DRIVER_MIX3_PACK(cpu, ndim, K_BATCH, pack1, pack2, pack3, n_pack1, n_pack2, n_pack3, unroll_m) \
  DRIVER_MIX3_PACK_SERIAL(cpu, ndim, K_BATCH, pack1, pack2, pack3, n_pack1, n_pack2, n_pack3, unroll_m)\
  DRIVER_MIX3_PACK_OMP(cpu, ndim, K_BATCH, pack1, pack2, pack3, n_pack1, n_pack2, n_pack3, unroll_m)

#endif
#endif

