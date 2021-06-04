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
 * File:        CommonDriver.h
 * Description: Common driver functions for GEMM operation. A driver function
 *              does blocking and calls packing/kernel/skinny_kernel functions
 *              to perform efficient matrix multiplication
 *****************************************************************************/

#include "ExpandMacro.h"
#include "CommonSched.h"
#include <string.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdlib.h>
#ifndef EMLL_SERIAL_ONLY
#include <omp.h>
#endif

#ifndef INCLUDE_COMMON_DRIVER
#define INCLUDE_COMMON_DRIVER

#define SKINNY_FUNC_BASE(FUNCTITLE, VOIDTITLE, ...) VOIDTITLE

#define SKINNY_FUNC_LIST_ITEM(NO, FUNCTITLE, VOIDTITLE, ...)\
  ,FUNCTITLE##NO##__VA_ARGS__

#define SKINNY_GEMM_FUNC_LIST(NUM, FUNCTITLE, VOIDTITLE, ...)\
  MACRO_EXP_##NUM(SKINNY_FUNC_BASE, SKINNY_FUNC_LIST_ITEM,\
    FUNCTITLE, VOIDTITLE, ##__VA_ARGS__)

/* blocking parameters */

#ifndef GEMM_R_MN
#define GEMM_R_MN 1024
#endif

#ifndef GEMM_D_MN
#define GEMM_D_MN 192
#endif

#ifndef GEMM_D_K
#define GEMM_D_K 192
#endif

/* GEMM_D_K * GEMM_UNROLL_M or GEMM_D_K * GEMM_UNROLL_N fit in L1 cache */
/* GEMM_D_K * GEMM_D_MN fit in L2 cache */
/* GEMM_R_MN is the last to optimize, not crucial to performance */

#if GEMM_D_MN > GEMM_R_MN
#define GEMM_S_MN GEMM_D_MN
#else
#define GEMM_S_MN GEMM_R_MN
#endif

#ifndef SCRATCH_K_CORD
#define SCRATCH_K_CORD(k) (k)
#endif

#define SCRATCH_GEMM_D_K (SCRATCH_K_CORD(GEMM_D_K - 1) + 1)

#define GEMM_STATIC_BUFFER(gemmtype, sbtype, satype) \
__thread __attribute__((aligned(4096))) satype\
  blas_##gemmtype##_sa[GEMM_S_MN * SCRATCH_GEMM_D_K];\
__thread __attribute__((aligned(4096))) sbtype\
  blas_##gemmtype##_sb[GEMM_S_MN * SCRATCH_GEMM_D_K];

/* serial driver function with packing both source matrices,
 * loop order: N { M { K } } */
#define GEMM_SERIAL_FUNC_LM(gemmtype, atype, satype, btype, sbtype, ctype,\
  unroll_m, unroll_n) \
static void gemmtype##_serial_lm_m##unroll_m##n##unroll_n(\
  int a_rowmajor, int b_rowmajor,\
  const atype *A, const btype *B, ctype *C,\
  uint32_t M, uint32_t N, uint32_t K, ctype beta_inp) {\
\
  satype * const sa = blas_##gemmtype##_sa;\
  sbtype * const sb = blas_##gemmtype##_sb;\
\
  uint32_t m_pos, n_pos, k_pos, m_inc, n_inc, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= (GEMM_D_K << 1)) k_inc = GEMM_D_K;\
    else if (k_inc > GEMM_D_K) k_inc >>= 1;\
    ctype beta = (k_pos == 0) ? beta_inp : 1;\
    for (n_pos = 0; n_pos < N; n_pos += n_inc) {\
      n_inc = N - n_pos;\
      if (n_inc >= (GEMM_R_MN << 1)) n_inc = GEMM_R_MN;\
      else if (n_inc > GEMM_R_MN) n_inc >>= 1;\
      if (b_rowmajor) {\
        gemmtype##_##btype##_##sbtype##_tcopy_unroll##unroll_n(\
          B + k_pos * N + n_pos, sb, N, n_inc, k_inc);\
      } else {\
        gemmtype##_##btype##_##sbtype##_ncopy_unroll##unroll_n(\
          B + n_pos * K + k_pos, sb, K, k_inc, n_inc);\
      }\
      for (m_pos = 0; m_pos < M; m_pos += m_inc) {\
        m_inc = M - m_pos;\
        if (m_inc > GEMM_D_MN) m_inc = GEMM_D_MN;\
        if (a_rowmajor) {\
          gemmtype##_##atype##_##satype##_ncopy_unroll##unroll_m(\
            A + m_pos * K + k_pos, sa, K, k_inc, m_inc);\
        } else {\
          gemmtype##_##atype##_##satype##_tcopy_unroll##unroll_m(\
            A + k_pos * M + m_pos, sa, M, m_inc, k_inc);\
        }\
        uint32_t scratch_k_inc = (k_inc == 0) ? 0 :\
          SCRATCH_K_CORD(k_inc - 1) + 1;\
        gemmtype##_kernel_lm_m##unroll_m##n##unroll_n(m_inc, n_inc,\
          scratch_k_inc,\
          beta, sa, sb, C + n_pos * M + m_pos, M);\
      }\
    }\
  }\
}

/* serial driver function with packing both source matrices,
 * loop order: M { N { K } } */
#define GEMM_SERIAL_FUNC_LN(gemmtype, atype, satype, btype, sbtype, ctype,\
  unroll_m, unroll_n) \
static void gemmtype##_serial_ln_m##unroll_m##n##unroll_n(\
  int a_rowmajor, int b_rowmajor,\
  const atype *A, const btype *B, ctype *C,\
  uint32_t M, uint32_t N, uint32_t K, ctype beta_inp) {\
\
  satype * const sa = blas_##gemmtype##_sa;\
  sbtype * const sb = blas_##gemmtype##_sb;\
\
  uint32_t m_pos, n_pos, k_pos, m_inc, n_inc, k_inc;\
  for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
    k_inc = K - k_pos;\
    if (k_inc >= (GEMM_D_K << 1)) k_inc = GEMM_D_K;\
    else if (k_inc > GEMM_D_K) k_inc >>= 1;\
    ctype beta = (k_pos == 0) ? beta_inp : 1;\
    for (m_pos = 0; m_pos < M; m_pos += m_inc) {\
      m_inc = M - m_pos;\
      if (m_inc >= (GEMM_R_MN << 1)) m_inc = GEMM_R_MN;\
      else if (m_inc > GEMM_R_MN) m_inc >>= 1;\
      if (a_rowmajor) {\
        gemmtype##_##atype##_##satype##_ncopy_unroll##unroll_m(\
          A + m_pos * K + k_pos, sa, K, k_inc, m_inc);\
      } else {\
        gemmtype##_##atype##_##satype##_tcopy_unroll##unroll_m(\
          A + k_pos * M + m_pos, sa, M, m_inc, k_inc);\
      }\
      for (n_pos = 0; n_pos < N; n_pos += n_inc) {\
        n_inc = N - n_pos;\
        if (n_inc > GEMM_D_MN) n_inc = GEMM_D_MN;\
        if (b_rowmajor) {\
          gemmtype##_##btype##_##sbtype##_tcopy_unroll##unroll_n(\
            B + k_pos * N + n_pos, sb, N, n_inc, k_inc);\
        } else {\
          gemmtype##_##btype##_##sbtype##_ncopy_unroll##unroll_n(\
            B + n_pos * K + k_pos, sb, K, k_inc, n_inc);\
        }\
        uint32_t scratch_k_inc = (k_inc == 0) ? 0 :\
          SCRATCH_K_CORD(k_inc - 1) + 1;\
        gemmtype##_kernel_ln_m##unroll_m##n##unroll_n(m_inc, n_inc,\
          scratch_k_inc,\
          beta, sa, sb, C + n_pos * M + m_pos, M);\
      }\
    }\
  }\
}

/* inline function to check arguments */
static inline bool inline_gemm_par_valid(const void *A, const void *B,
  void *C, uint32_t M, uint32_t N, uint32_t K) {

  bool a_valid = A || (M == 0 || K == 0);
  bool b_valid = B || (N == 0 || K == 0);
  bool c_valid = C || (M == 0 || N == 0);

  return a_valid && b_valid && c_valid;
}

/* serial GEMM driver function */
#define GEMM_SERIAL_FUNC(gemmtype, atype, satype, btype, sbtype, ctype,\
  unroll_l2, unroll_l1, skin1_maxm, skin1_maxn, skin2_maxm, skin2_maxn, ...)\
\
GEMM_STATIC_BUFFER(gemmtype, sbtype, satype)\
\
GEMM_SERIAL_FUNC_LM(gemmtype, atype, satype, btype, sbtype, ctype,\
  unroll_l2, unroll_l1)\
\
GEMM_SERIAL_FUNC_LN(gemmtype, atype, satype, btype, sbtype, ctype,\
  unroll_l1, unroll_l2)\
\
static void arowmajor_bskinny_void(\
  const atype *A_mat, const btype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order, ctype beta_inp) { return; }\
\
static void bcolmajor_askinny_void(\
  const btype *A_mat, const atype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order, ctype beta_inp) { return; }\
\
static void acolmajor_bskinny_void(\
  const atype *A_mat, const btype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order, ctype beta_inp) { return; }\
\
static void browmajor_askinny_void(\
  const btype *A_mat, const atype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order, ctype beta_inp) { return; }\
\
static void (* gemmtype##_bskinny1[]) (\
  const atype *A_mat, const btype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order, ctype beta_inp) = {\
  SKINNY_GEMM_FUNC_LIST(skin1_maxn,\
    gemmtype##_arowmajor_bskinny_a##atype##_b##btype##_n,\
    arowmajor_bskinny_void) };\
\
static void (* gemmtype##_askinny1[]) (\
  const btype *A_mat, const atype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order, ctype beta_inp) = {\
  SKINNY_GEMM_FUNC_LIST(skin1_maxm,\
    gemmtype##_arowmajor_bskinny_a##btype##_b##atype##_n,\
    bcolmajor_askinny_void) };\
\
static void (* gemmtype##_bskinny2[]) (\
  const atype *A_mat, const btype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order, ctype beta_inp) = {\
  SKINNY_GEMM_FUNC_LIST(skin2_maxn,\
    gemmtype##_acolmajor_bskinny_a##atype##_b##btype##_n,\
    acolmajor_bskinny_void) };\
\
static void (* gemmtype##_askinny2[]) (\
  const btype *A_mat, const atype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order, ctype beta_inp) = {\
  SKINNY_GEMM_FUNC_LIST(skin2_maxm,\
    gemmtype##_acolmajor_bskinny_a##btype##_b##atype##_n,\
    browmajor_askinny_void) };\
\
int gemmtype##_serial(int a_rowmajor, int b_rowmajor,\
  const atype *A, const btype *B, ctype *C,\
  uint32_t M, uint32_t N, uint32_t K, ctype beta_inp) {\
\
  if (!inline_gemm_par_valid(A, B, C, M, N, K)) return 1;\
  if (0 __VA_ARGS__) return 2;\
\
  if (K == 0) {\
    if (beta_inp != (ctype)1.0) {\
      const uint64_t MN = (uint64_t)M * (uint64_t)N;\
      for (uint64_t pos = 0; pos < MN; ++pos) {\
        C[pos] *= beta_inp;\
      }\
    }\
    return 0;\
  }\
\
  if (N <= skin1_maxn && a_rowmajor) {\
    (* gemmtype##_bskinny1[N])(A, B, C, M, K, b_rowmajor ? 1 : 0, beta_inp);\
    return 0;\
  }\
  if (M <= skin1_maxm && !b_rowmajor) {\
    (* gemmtype##_askinny1[M])(B, A, C, N, K, a_rowmajor ? 2 : 3, beta_inp);\
    return 0;\
  }\
  if (N <= skin2_maxn && !a_rowmajor) {\
    (* gemmtype##_bskinny2[N])(A, B, C, M, K, b_rowmajor ? 1 : 0, beta_inp);\
    return 0;\
  }\
  if (M <= skin2_maxm && b_rowmajor) {\
    (* gemmtype##_askinny2[M])(B, A, C, N, K, a_rowmajor ? 2 : 3, beta_inp);\
    return 0;\
  }\
\
  if ((N >> 1) > M) {\
    gemmtype##_serial_ln_m##unroll_l1##n##unroll_l2(\
      a_rowmajor, b_rowmajor, A, B, C, M, N, K, beta_inp);\
  } else {\
    gemmtype##_serial_lm_m##unroll_l2##n##unroll_l1(\
      a_rowmajor, b_rowmajor, A, B, C, M, N, K, beta_inp);\
  }\
  return 0;\
}

#ifdef EMLL_SERIAL_ONLY

#define GEMM_PARALLEL_FUNC(gemmtype, atype, satype, btype, sbtype, ctype,\
  unroll_l2, unroll_l1, skin1_maxm, skin1_maxn, skin2_maxm, skin2_maxn, ...) \
\
GEMM_SERIAL_FUNC(gemmtype, atype, satype, btype, sbtype, ctype,\
  unroll_l2, unroll_l1, skin1_maxm, skin1_maxn, skin2_maxm, skin2_maxn, ##__VA_ARGS__)\
int gemmtype(int a_rowmajor, int b_rowmajor,\
  const atype *A, const btype *B, ctype *C,\
  uint32_t M, uint32_t N, uint32_t K, ctype beta_inp, uint32_t num_threads) {\
\
  return gemmtype##_serial(a_rowmajor, b_rowmajor, A, B, C, M, N, K, beta_inp);\
}

#else

/* OpenMP GEMM driver function */
#define GEMM_PARALLEL_FUNC(gemmtype, atype, satype, btype, sbtype, ctype,\
  unroll_l2, unroll_l1, skin1_maxm, skin1_maxn, skin2_maxm, skin2_maxn, ...) \
\
GEMM_SERIAL_FUNC(gemmtype, atype, satype, btype, sbtype, ctype,\
  unroll_l2, unroll_l1, skin1_maxm, skin1_maxn, skin2_maxm, skin2_maxn, ##__VA_ARGS__)\
\
static void arowmajor_bskinny_void_omp(\
  const atype *A_mat, const btype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  ctype beta_inp, uint32_t num_threads) { return; }\
\
static void bcolmajor_askinny_void_omp(\
  const btype *A_mat, const atype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  ctype beta_inp, uint32_t num_threads) { return; }\
\
static void acolmajor_bskinny_void_omp(\
  const atype *A_mat, const btype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  ctype beta_inp, uint32_t num_threads) { return; }\
\
static void browmajor_askinny_void_omp(\
  const btype *A_mat, const atype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  ctype beta_inp, uint32_t num_threads) { return; }\
\
static void (* gemmtype##_bskinny1_omp[]) (\
  const atype *A_mat, const btype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  ctype beta_inp, uint32_t num_threads) = {\
  SKINNY_GEMM_FUNC_LIST(skin1_maxn,\
    gemmtype##_arowmajor_bskinny_a##atype##_b##btype##_n,\
    arowmajor_bskinny_void_omp, _omp) };\
\
static void (* gemmtype##_askinny1_omp[]) (\
  const btype *A_mat, const atype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  ctype beta_inp, uint32_t num_threads) = {\
  SKINNY_GEMM_FUNC_LIST(skin1_maxm,\
    gemmtype##_arowmajor_bskinny_a##btype##_b##atype##_n,\
    bcolmajor_askinny_void_omp, _omp) };\
\
static void (* gemmtype##_bskinny2_omp[]) (\
  const atype *A_mat, const btype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  ctype beta_inp, uint32_t num_threads) = {\
  SKINNY_GEMM_FUNC_LIST(skin2_maxn,\
    gemmtype##_acolmajor_bskinny_a##atype##_b##btype##_n,\
    acolmajor_bskinny_void_omp, _omp) };\
\
static void (* gemmtype##_askinny2_omp[]) (\
  const btype *A_mat, const atype *B_skin, ctype *C_skin,\
  uint32_t M, uint32_t K, uint8_t b_c_order,\
  ctype beta_inp, uint32_t num_threads) = {\
  SKINNY_GEMM_FUNC_LIST(skin2_maxm,\
    gemmtype##_acolmajor_bskinny_a##btype##_b##atype##_n,\
    browmajor_askinny_void_omp, _omp) };\
\
int gemmtype(int a_rowmajor, int b_rowmajor,\
  const atype *A, const btype *B, ctype *C,\
  uint32_t M, uint32_t N, uint32_t K, ctype beta_inp, uint32_t num_threads) {\
\
  uint32_t rec_threads = (uint64_t)M * (uint64_t)N * (uint64_t)K \
    / (GEMM_D_K * GEMM_D_MN * unroll_l1) + 1;\
  if (num_threads > rec_threads) num_threads = rec_threads;\
  if (!num_threads) num_threads = rec_threads;\
  uint32_t max_threads = omp_get_max_threads();\
  if (num_threads > max_threads) num_threads = max_threads;\
\
  if (num_threads == 1 || K == 0) {\
    return gemmtype##_serial(a_rowmajor, b_rowmajor, A, B, C, M, N, K, beta_inp);\
  }\
\
  if (!inline_gemm_par_valid(A, B, C, M, N, K)) return 1;\
  if (0 __VA_ARGS__) return 2;\
\
  omp_set_num_threads(num_threads);\
  if (N <= skin1_maxn && a_rowmajor) {\
    (* gemmtype##_bskinny1_omp[N])(\
      A, B, C, M, K, b_rowmajor ? 1 : 0, beta_inp, num_threads);\
    return 0;\
  }\
  if (M <= skin1_maxm && !b_rowmajor) {\
    (* gemmtype##_askinny1_omp[M])(\
      B, A, C, N, K, a_rowmajor ? 2 : 3, beta_inp, num_threads);\
    return 0;\
  }\
  if (N <= skin2_maxn && !a_rowmajor) {\
    (* gemmtype##_bskinny2_omp[N])(\
      A, B, C, M, K, b_rowmajor ? 1 : 0, beta_inp, num_threads);\
    return 0;\
  }\
  if (M <= skin2_maxm && b_rowmajor) {\
    (* gemmtype##_askinny2_omp[M])(\
      B, A, C, N, K, a_rowmajor ? 2 : 3, beta_inp, num_threads);\
    return 0;\
  }\
\
  satype * const blas_master_sa = blas_##gemmtype##_sa;\
  sbtype * const blas_master_sb = blas_##gemmtype##_sb;\
  uint32_t acopy_dim_left, bcopy_dim_left;\
  uint64_t mn_task_end;\
\
  _Pragma("omp parallel")\
  {\
    const uint32_t tid = omp_get_thread_num();\
    uint32_t k_pos, m_pos, n_pos, k_inc, m_inc, n_inc;\
    uint32_t bcopy_dim_start, bcopy_dim_end, acopy_dim_start, acopy_dim_end;\
    uint32_t gemm_mstart, gemm_mend, gemm_nstart, gemm_nend;\
    uint64_t gemm_mn_max;\
    for (k_pos = 0; k_pos < K; k_pos += k_inc) {\
      k_inc = K - k_pos;\
      if (k_inc >= (GEMM_D_K << 1)) k_inc = GEMM_D_K;\
      else if (k_inc > GEMM_D_K) k_inc >>= 1;\
      const uint32_t scratch_k_inc = (k_inc == 0) ? 0 :\
        SCRATCH_K_CORD(k_inc - 1) + 1;\
      const ctype beta = (k_pos == 0) ? beta_inp : 1;\
      for (n_pos = 0; n_pos < N; n_pos += n_inc) {\
        n_inc = N - n_pos;\
        if (n_inc >= (GEMM_R_MN << 1)) n_inc = GEMM_R_MN;\
        else if (n_inc > GEMM_R_MN) n_inc >>= 1;\
\
        if (!tid) bcopy_dim_left = n_inc;\
        _Pragma("omp barrier");\
\
        while (get_copy_task(&bcopy_dim_left, unroll_l1 << 3,\
          &bcopy_dim_start, &bcopy_dim_end)) {\
          if (b_rowmajor) {\
            gemmtype##_##btype##_##sbtype##_tcopy_unroll##unroll_l1(\
              B + k_pos * N + n_pos + bcopy_dim_start,\
              blas_master_sb + bcopy_dim_start * scratch_k_inc,\
              N, bcopy_dim_end - bcopy_dim_start, k_inc);\
          } else {\
            gemmtype##_##btype##_##sbtype##_ncopy_unroll##unroll_l1(\
              B + K * (n_pos + bcopy_dim_start) + k_pos,\
              blas_master_sb + bcopy_dim_start * scratch_k_inc,\
              K, k_inc, bcopy_dim_end - bcopy_dim_start);\
          }\
        }\
\
        for (m_pos = 0; m_pos < M; m_pos += m_inc) {\
          m_inc = M - m_pos;\
          if (m_inc >= (GEMM_R_MN << 1)) m_inc = GEMM_R_MN;\
          else if (m_inc > GEMM_R_MN) m_inc >>= 1;\
\
          if (!tid) acopy_dim_left = m_inc;\
          _Pragma("omp barrier");\
\
          while (get_copy_task(&acopy_dim_left, unroll_l2 << 3,\
            &acopy_dim_start, &acopy_dim_end)) {\
            if (a_rowmajor) {\
              gemmtype##_##atype##_##satype##_ncopy_unroll##unroll_l2(\
                A + K * (m_pos + acopy_dim_start) + k_pos,\
                blas_master_sa + acopy_dim_start * scratch_k_inc,\
                K, k_inc, acopy_dim_end - acopy_dim_start);\
            } else {\
              gemmtype##_##atype##_##satype##_tcopy_unroll##unroll_l2(\
                A + M * k_pos + m_pos + acopy_dim_start,\
                blas_master_sa + acopy_dim_start * scratch_k_inc,\
                M, acopy_dim_end - acopy_dim_start, k_inc);\
            }\
          }\
\
          if (!tid) mn_task_end = (uint64_t)n_pos << 32 | (uint64_t)m_pos;\
          gemm_mn_max = ((uint64_t)(n_pos + n_inc) << 32)\
            | (uint64_t)(m_pos + m_inc);\
          _Pragma("omp barrier");\
\
          while (get_mn_task(&mn_task_end,\
            &gemm_mstart, &gemm_nstart, &gemm_mend, &gemm_nend,\
            ((uint64_t)unroll_l1 << 32) | ((GEMM_D_MN >> 2) / unroll_l2 * unroll_l2),\
            GEMM_D_MN, n_pos, gemm_mn_max, num_threads)) {\
\
            gemmtype##_kernel_lm_m##unroll_l2##n##unroll_l1(\
              gemm_mend - gemm_mstart, gemm_nend - gemm_nstart, scratch_k_inc, beta,\
              blas_master_sa + (gemm_mstart - m_pos) * scratch_k_inc,\
              blas_master_sb + (gemm_nstart - n_pos) * scratch_k_inc,\
              C + M * gemm_nstart + gemm_mstart, M);\
          }\
        }\
      }\
    }\
  }\
  return 0;\
}

#endif

#endif
